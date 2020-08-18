import torch as t
from torch import nn

from .teacher import teacher
#except:from model.teacher import teacher
from .fastSal import fastsal
#except:from model.fastSal import student

def compute_hint_loss_l(student_d, teacher_d, batch_size, loss_func, layers, normalize=False):
    all_hint_loss_d = []
    count = 0
    for s_d, t_d in zip(student_d, teacher_d):
        l_tmp = loss_func(s_d.view(batch_size, -1), t_d.detach().view(batch_size, -1))
        if count in layers:
            if normalize:
                all_hint_loss_d.append(1 - t.mean(l_tmp))
            else:
                all_hint_loss_d.append(l_tmp)
        count += 1
    return sum(all_hint_loss_d)

def compute_log_prob_3losses(pred, gt, batch_size):
    pred = pred.view(batch_size, -1)
    gt = gt.view(batch_size, -1)
    gt_min = gt.min(1, keepdim=True)[0]
    gt_max = gt.max(1, keepdim=True)[0]
    gt_norm = (gt-gt_min)/(gt_max-gt_min)

    pred_kl = nn.functional.log_softmax(pred, dim=1)
    loss_kl = t.nn.functional.kl_div(pred_kl, gt, reduction='sum') / batch_size
    pred_cc = nn.functional.softmax(pred, dim=1)
    loss_cc = 1 - t.mean(t.nn.functional.cosine_similarity(pred_cc, gt))
    pred_bce = t.sigmoid(pred)
    loss_bce = t.nn.functional.binary_cross_entropy(pred_bce, gt_norm,reduction='mean')
    loss = 0.5*loss_kl + 0.3*loss_bce + 0.2*loss_cc
    return loss

class salgan_teacher_student(nn.Module):
    def __init__(self, pretrain_mode, model_type, path=None, state_dict=None,
                 use_gt=False, use_teacher=False, use_pseudo_gt=False, use_probability_gt=False):
        super(salgan_teacher_student, self).__init__()
        if pretrain_mode:
            use_teacher = True
        if use_teacher:
            self.salgan_teacher = teacher(path).eval()
        self.student_net = fastsal(pretrain_mode, model_type, state_dict)

        self.pretrain_mode = pretrain_mode
        self.use_teacher = use_teacher
        self.use_gt = use_gt
        self.use_probability_gt = use_probability_gt
        self.use_pseudo_gt = use_pseudo_gt
        self.model_type = model_type

        self.bce = nn.BCELoss(reduction='none')
        self.bce_r = nn.BCELoss()
        self.bcel = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss(reduction='mean')
        self.cosine_sim = nn.CosineSimilarity()
        self.kl = nn.KLDivLoss()
        self.avg_pool = nn.AvgPool2d((4,4))
        self.sigmoid = nn.Sigmoid()

    def remove_teacher(self):
        del self.salgan_teacher

    def forward(self, inputs, vgg_inputs, targets=None):
        batch_size = vgg_inputs.shape[0]
        if targets is not None:
            if isinstance(targets, list):
                #fixation_target = targets[1]
                sal_target = targets[0]
            else:
                sal_target = targets

        student_out = self.student_net.forward(vgg_inputs)
        #print(self.student_net.pretrain_mode)
        #print(len(student_out))
        loss = []
        if self.use_teacher:
            #print('use teacher','!!')
            teacher_sal, teacher_code, teacher_e, teacher_d = self.salgan_teacher(inputs)
            if self.pretrain_mode:
                loss.append(self.compute_pretrain_loss(student_out, teacher_d, batch_size))
            else:
                if self.training and self.use_gt or not self.use_gt:
                    #print('except use gt and eval')
                    student_out_sigmoid = self.sigmoid(student_out)
                    l_st_sal = self.bce_r(student_out_sigmoid.view(batch_size, -1), teacher_sal.detach().view(batch_size, -1))
                    loss.append(l_st_sal)
        if self.use_gt:
            #print('use gt')
            student_out_sigmoid = self.sigmoid(student_out)
            loss.append(self.bce_r(student_out_sigmoid.view(batch_size, -1), sal_target.view(batch_size, -1)))
        if self.use_pseudo_gt:
            if self.use_probability_gt:
                #print('use probability gt')
                loss.append(compute_log_prob_3losses(student_out, sal_target, batch_size))
        #print(loss,'!!!!')
        return sum(loss)

    def compute_pretrain_loss(self, student_d, teacher_d, batch_size):
        if self.model_type is 'C':
            l_hint_d = compute_hint_loss_l(student_d, teacher_d[::-1], batch_size, self.mse, [0, 1, 2, 3])
            return l_hint_d
        else:
            exit()

    def predict(self, inputs):
        salmap = self.student_net(inputs)
        if not self.use_probability_gt:
            salmap = self.sigmoid(salmap)
        #salmap = nn.functional.interpolate(salmap, size=(480, 640), align_corners=True, mode='bilinear')
        return salmap

    def get_optimizer(self, lr, use_adam=False, exclude_list=[None]):
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad and key.split('.')[1] not in exclude_list:
                #print(key)
                if 'bias' in key:
                    if 'ecer' in key:
                        params += [{'params': [value], 'lr': lr * 1, 'weight_decay': 0}]
                    else:
                        params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    if 'ecer' in key:
                        params += [{'params': [value], 'lr': lr * 0.5, 'weight_decay': 0.0005}]
                    else:
                        params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]

        if use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, lr=lr, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer


if __name__ == '__main__':
    test_tensor = t.rand(1,3,192,256)
    target_tensor = t.ones(1, 1, 192, 256)/(192*256)
    path = '/home/feiyan/data/Github/SAL_compress/salgan_pytorch/trained_models/' \
           'salgan_adversarial2/models/gen_42.pt'
    sts = salgan_teacher_student('fine_tune_deepgaze2_3losses', 'C', path)
    y = sts.forward(test_tensor, test_tensor, [target_tensor, target_tensor])
    print(y.shape)
    print(y.item())