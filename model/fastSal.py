import torch.nn as nn
import torch as t

from .utils import get_student_features
from .fastsal_generator import generate_model9, generate_model10


class fastsal(nn.Module):
    def __init__(self, pretrain_mode, model_type, state_dict=None):
        super(fastsal, self).__init__()
        self.model_type = model_type
        self.pretrain_mode = pretrain_mode

        if model_type == 'C':
            if pretrain_mode:
                self.ecer, self.adaptation_layer = generate_model9(pretrain_mode, forward_hook_index=range(1,19,1))
            else:
                self.ecer, self.adaptation_layer, self.comb = \
                    generate_model9(pretrain_mode, student_model=self, state_dict=state_dict, forward_hook_index=range(1,19,1))
            self.shuffle_up2 = nn.PixelShuffle(2)
        if model_type == 'A':
            if pretrain_mode:
                self.ecer, self.adaptation_layer = generate_model10(pretrain_mode, forward_hook_index=range(1,19,1))
            else:
                self.ecer, self.adaptation_layer,self.adp1, self.adp2,self.adp3,self.adp4, self.comb = \
                    generate_model10(pretrain_mode, student_model=self, state_dict=state_dict, forward_hook_index=range(1,19,1))
            self.shuffle_up2 = nn.PixelShuffle(2)

    def init_pretrain(self):
        self.__init__(True, self.model_type)
    def set_grad(self, exclude_list):
        for name, param in self.named_parameters():
            m = name.split('.')[0]
            if m in exclude_list:
                print(name, 'set grad to False')
                param.requires_grad = False

    def features(self, vgg_inputs):
        student_code = self.ecer(vgg_inputs)
        h = [(0, 1), (1, 3), (3, 6), (6, 13), (13, 18)]
        student_e = get_student_features(range(0, 18, 1), 'student_encoder')
        student_e0 = student_e[0]
        student_e1 = self.shuffle_up2(t.cat(student_e[h[1][0]:h[1][1]], dim=1))
        student_e1 = t.cat([student_e0, student_e1], dim=1)
        student_e2 = self.shuffle_up2(t.cat(student_e[h[2][0]:h[2][1]], dim=1))
        student_e3 = self.shuffle_up2(t.cat(student_e[h[3][0]:h[3][1]], dim=1))
        student_e4 = self.shuffle_up2(t.cat(student_e[h[4][0]:h[4][1]], dim=1))
        student_e = [student_e1, student_e2, student_e3, student_e4]
        student_d = self.adaptation_layer(student_e)
        return student_d

    def forward_pretrain(self, vgg_inputs):
        if self.model_type is 'C':
            return self.features(vgg_inputs)
        elif self.model_type is 'A':
            print('Addition version does not support pretrain since the miss match of the '
                  'channel of adaptation layer.')
            exit()

    def forward_fine_tune(self, vgg_inputs, return_hint=False):
        student_d = self.features(vgg_inputs)
        if self.model_type is 'C':
            sal_branch = self.comb([t.cat(student_d, dim=1)])[0]
            return sal_branch
        elif self.model_type is 'A':
            f4 = self.shuffle_up2(self.adp4(student_d[3]))
            f3 = self.shuffle_up2(self.adp3(student_d[2], f4))
            f2 = self.shuffle_up2(self.adp2(student_d[1], f3))
            f1 = self.shuffle_up2(self.adp1(student_d[0], f2))
            out = self.comb([f1])[0]
            return out

    def forward(self, vgg_inputs):
        if self.pretrain_mode:
            return self.forward_pretrain(vgg_inputs)
        else:
            return self.forward_fine_tune(vgg_inputs)

if __name__ == '__main__':
    m = fastsal('pretrain', 'A')
    x = t.zeros((10, 3, 192, 256))
    y = m(x)
    print(len(y))