import torch as t
import torch.nn as nn
from torchvision.models import mobilenet_v2

from .salgan_generator import create_model

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

def decom_salgan(path):
    model = create_model()
    model.load_state_dict(t.load(path)['state_dict'])
    model = list(model)
    #for i,x in enumerate(model):
    #    print(i,x)
    model[61].register_forward_hook(get_activation('sal_gan_salimg'))
    model[54].register_forward_hook(get_activation('sal_gan_d0'))
    model[49].register_forward_hook(get_activation('sal_gan_d1'))
    model[42].register_forward_hook(get_activation('sal_gan_d2'))
    model[35].register_forward_hook(get_activation('sal_gan_d3'))
    model[29].register_forward_hook(get_activation('sal_gan_code'))
    model[22].register_forward_hook(get_activation('sal_gan_e0'))
    model[15].register_forward_hook(get_activation('sal_gan_e1'))
    model[8].register_forward_hook(get_activation('sal_gan_e2'))
    model[3].register_forward_hook(get_activation('sal_gan_e3'))
    model = nn.Sequential(*model)
    for param in model.parameters():
        param.requires_grad = False
    #model.eval()
    #model.cuda()
    return model

def register_layers(model, name_list, prefix):
    for i, idx in enumerate(name_list):
        model[idx].register_forward_hook(get_activation(prefix+'_{}'.format(i)))
    return model
def get_student_features(name_list, prefix):
    data = []
    for name in name_list:
        data.append(activation[prefix+'_{}'.format(name)])
    return data
def get_teacher_supervision(inputs, salgan_teacher):
    teacher_sal = salgan_teacher(inputs)
    teacher_code = activation['sal_gan_code']
    teacher_e0 = activation['sal_gan_e3']
    teacher_e1 = activation['sal_gan_e2']
    teacher_e2 = activation['sal_gan_e1']
    teacher_e3 = activation['sal_gan_e0']
    teacher_d0 = activation['sal_gan_d3']
    teacher_d1 = activation['sal_gan_d2']
    teacher_d2 = activation['sal_gan_d1']
    teacher_d3 = activation['sal_gan_d0']
    # print('teacher', teacher_sal.shape, teacher_code.shape)
    # print('intermediate', teacher_e0.shape, teacher_e1.shape, teacher_e2.shape, teacher_e3.shape)
    # print('intermediate', teacher_d0.shape, teacher_d1.shape, teacher_d2.shape, teacher_d3.shape)
    return teacher_sal, teacher_code, [teacher_e0, teacher_e1, teacher_e2, teacher_e3], \
           [teacher_d0, teacher_d1, teacher_d2, teacher_d3]

def mobilenetv2_pretrain(forward_hook_index=None):
    model = mobilenet_v2(pretrained=True)
    features = list(model.features)
    #for i, x in enumerate(features): print(i, x)
    #exit()
    if forward_hook_index is not None:
        register_layers(features, forward_hook_index, 'student_encoder')
    features = nn.Sequential(*features)
    return features