import model.fastSal as fastsal
from utils import load_weight
import torch

if __name__ == '__main__':
    coco_c = 'weights/coco_C.pth'  # coco_C
    coco_a = 'weights/coco_A.pth'  # coco_A
    salicon_c = 'weights/salicon_C.pth'  # salicon_C
    salicon_a = 'weights/salicon_A.pth'  # coco_A

    x = torch.zeros((10, 3, 192, 256))

    model = fastsal.fastsal(pretrain_mode=False, model_type='A')
    state_dict, opt_state = load_weight(coco_a, remove_decoder=False)
    model.load_state_dict(state_dict)
    y = model(x)
    print(y.shape)

    model = fastsal.fastsal(pretrain_mode=False, model_type='A')
    state_dict, opt_state = load_weight(salicon_a, remove_decoder=False)
    model.load_state_dict(state_dict)
    y = model(x)
    print(y.shape)

    model = fastsal.fastsal(pretrain_mode=False, model_type='C')
    state_dict, opt_state = load_weight(coco_c, remove_decoder=False)
    model.load_state_dict(state_dict)
    y = model(x)
    print(y.shape)

    model = fastsal.fastsal(pretrain_mode=False, model_type='C')
    state_dict, opt_state = load_weight(salicon_c, remove_decoder=False)
    model.load_state_dict(state_dict)
    y = model(x)
    print(y.shape)

    print('All model loaded and tested')