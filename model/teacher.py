import torch.nn as nn
import torch as t

from .utils import decom_salgan, get_teacher_supervision


class teacher(nn.Module):
    def __init__(self, path):
        super(teacher, self).__init__()
        self.salgan_teacher = decom_salgan(path)

    def forward(self, inputs):
        #batch_size = inputs.shape[0]
        teacher_sal, teacher_code, teacher_e, teacher_d = get_teacher_supervision(inputs, self.salgan_teacher)
        # print(teacher_d[0].shape, teacher_d[1].shape, teacher_d[2].shape, teacher_d[3].shape)
        #print(student_e[0].shape, student_e[1].shape, student_e[2].shape, student_e[-1].shape)
        return teacher_sal, teacher_code, teacher_e, teacher_d
    def add_dropout(self, p=0.5):
        insert_index = [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29, 31, 33, 35, 38, 40, 42, 45, 47, 49,
                        52, 54, 57, 59]
        insert_index = [i+1 for i in insert_index]
        salgan_dropout = list(self.salgan_teacher)
        acc = 0
        for i in insert_index:
            salgan_dropout.insert(i + acc, nn.Dropout2d(p=p))
            acc += 1
        self.salgan_teacher = nn.Sequential(*salgan_dropout)
        #for i, x in enumerate(list(self.salgan_teacher)): print(i, x)
        #exit()
    def set_require_grad(self, require_grad=False):
        for param in self.salgan_teacher.parameters():
            param.requires_grad = require_grad

if __name__ == '__main__':
    test_tensor = t.ones(4, 3, 192, 256)
    target_tensor = t.zeros(4, 1, 192, 256)
    sts = teacher('/home/feiyan/data/Github/SAL_compress/salgan_pytorch/trained_models/'
                  'salgan_adversarial2/models/gen_42.pt')
    sts.add_dropout()
    sts.train()
    #sts.eval()
    y,_,_,_ = sts.forward(test_tensor)
    print(y.shape)
    print(y)