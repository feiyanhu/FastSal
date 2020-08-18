import numpy as np
import torch as t
from torch.utils.data import DataLoader, ConcatDataset
import argparse
from os.path import exists, join
from os import mkdir

import dataset.dataset as dataset
import dataset.coco as coco
from model import student_teacher
from utils import save_weight, load_weight

def salicon_data(batch_size, dataset_path):
    ds_train = dataset.Salicon(dataset_path, mode='train')
    ds_validate = dataset.Salicon(dataset_path, mode='val', N=1000)
    ds_metric = dataset.Salicon(dataset_path, mode='val', type=('vgg_img', 'sal_img', 'fixation_path'), N=100)

    dataloader = {
        'train': DataLoader(ds_train, batch_size=batch_size,
                            shuffle=True, num_workers=4),
        'val': DataLoader(ds_validate, batch_size=int(batch_size/2),
                          shuffle=False, num_workers=4),
        'metric': DataLoader(ds_metric, batch_size=int(batch_size/2),
                             shuffle=False, num_workers=4),
    }
    return dataloader


def coco_data(batch_size, dataset_path, input_type=['deep_gaze_II'], pseudo_path=None):
    size = (192, 256)
    ds_train = coco.COCO(dataset_path, mode='train', size=size, type=input_type+['vgg_img'])
    ds_train.set_pseudo_gt_path(pseudo_path)
    ds_test = coco.COCO(dataset_path, mode='test', size=size, type=input_type+['vgg_img'])
    ds_test.set_pseudo_gt_path(pseudo_path)
    ds_train = ConcatDataset([ds_train, ds_test])
    ds_validate = coco.COCO(dataset_path, mode='val', size=size, type=input_type+['vgg_img'])
    ds_validate.set_pseudo_gt_path(pseudo_path)
    dataloader = {
        'train': DataLoader(ds_train, batch_size=batch_size,
                            shuffle=True, num_workers=4),
        'val': DataLoader(ds_validate, batch_size=int(batch_size/4),
                          shuffle=False, num_workers=4),
    }
    return dataloader


def train_one(model, dataloader, optimizer, mode,
              use_gt=False, use_pseudo_gt=False, use_teacher=False):
    all_loss = []
    for i, X in enumerate(dataloader[mode]):
        optimizer.zero_grad()
        inputs = None; gt_maps_input = None
        vgg_inputs = X['vgg_img'][0].cuda()
        if use_teacher:
            inputs = X['caffe_img'].cuda()
        if use_gt:
            gt_maps_input = X['sal_img'].cuda()
        if use_pseudo_gt:
            gt_maps_input = X['npy_img'].cuda()
        losses = model.forward(inputs, vgg_inputs, gt_maps_input)
        if mode == 'train':
            losses.backward()
            optimizer.step()
            all_loss.append(losses.item())
        elif mode == 'val':
            with t.no_grad():
                all_loss.append(losses.item())
        if i%10 == 0:
            print('{} current accumulated loss {}'.format(i, np.mean(all_loss)))
            #break
    return np.mean(all_loss), model


def start_train(model_type, batch_size, dataset_name, dataset_path, teacher_path, direct,
                model_name, pretrain_path=None, pseudo_path=None):
    if dataset_name == 'salicon':
        dataloader = salicon_data(batch_size, dataset_path)
        use_gt = True
        use_pseudo_gt = False
        use_teacher = True
        use_probability_gt = False
    elif dataset_name == 'coco':
        dataloader = coco_data(batch_size, dataset_path, input_type=['npy_img'], pseudo_path=pseudo_path)
        use_gt = False
        use_pseudo_gt = True
        use_teacher = False
        use_probability_gt = True

    if pretrain_path:
        state_dict, opt_state = load_weight(pretrain_path, remove_decoder=False)
    else:
        state_dict = None

    model = student_teacher.salgan_teacher_student(False, model_type, teacher_path, state_dict,
                                                   use_gt=use_gt, use_teacher=use_teacher,
                                                   use_probability_gt=use_probability_gt,
                                                   use_pseudo_gt=use_pseudo_gt)
    model.cuda()

    lr = 0.01
    lr_decay = 0.1
    optimizer = model.get_optimizer(lr)
    smallest_val = None
    best_epoch = None

    for epoch in range(0,100, 1):
        #with t.no_grad():
        #    metrics = get_saliency_metrics(dataloader['metric'], model, N=100)
        model.train()
        loss_train, model = train_one(model, dataloader, optimizer, 'train', use_gt=use_gt, use_teacher=use_teacher,
                                      use_pseudo_gt=use_pseudo_gt)
        print('{} loss train {}, lr {}'.format(epoch, loss_train, lr))
        print('--------------------------------------------->>>>>>')
        model.eval()
        loss_val, model = train_one(model, dataloader, optimizer, 'val', use_gt=use_gt, use_teacher=use_teacher,
                                    use_pseudo_gt=use_pseudo_gt)
        print('--------------------------------------------->>>>>>')
        print('{} loss val {}'.format(epoch, loss_val))

        smallest_val, best_epoch, model, optimizer = save_weight(smallest_val, best_epoch, loss_val, epoch,
                                                                 direct, model_name, model, optimizer)
        if epoch == 15 or epoch == 30 or epoch == 60:
            path = '{}/{}/{}_{:f}.pth'.format(direct, model_name, best_epoch, smallest_val)
            state_dict, opt_state = load_weight(path, remove_decoder=False)
            model.student_net.load_state_dict(state_dict)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            lr = lr * lr_decay

if __name__ == '__main__':
    path = '/home/feiyan/data/Github/SAL_compress/salgan_pytorch/trained_models/' \
           'salgan_adversarial2/models/gen_42.pt'
    salicon_pretrain_path = 'weights/salicon_salgan_pretrain.pth'  # salgan pretrain
    coco_pretrain_path = 'weights/coco_salgan_pretrain.pth' # coco pretrain

    coco_path = '/data/coco/'
    coco_pseudo_path = '/home/feiyan/data/generated_sal/deep_gaze_II_coco_with_cb_npy/'
    salicon_path = '/data/Datasets/SALICON/'

    parser = argparse.ArgumentParser(description='configs for pretrain.')
    parser.add_argument('-model_type', action='store', dest='model_type',
                        help='model type can be either C(oncatenation) or A(ddition)', default='A')
    parser.add_argument('-batch_size', action='store', dest='batch_size',
                        help='batch size', default=10, type=int)
    parser.add_argument('-dataset_name', action='store', dest='dataset_name',
                        help='dataset_name either coco or salicon', default='salicon')
    parser.add_argument('-dataset_path', action='store', dest='dataset_path',
                        help='path to dataset')
    parser.add_argument('-teacher_path', action='store', dest='teacher_path',
                        help='path to teacher weight (SALGAN)')
    parser.add_argument('-save_dir', action='store', dest='save_dir',
                        help='directory to save pretrained weights', default='checkpoint')
    parser.add_argument('-model_name', action='store', dest='model_name',
                        help='pretrained model name', default='salicon_salgan_finetune')
    parser.add_argument('-pretrain_path', action='store', dest='pretrain_path',
                        help='path to pretrained model', default='weights/salicon_salgan_pretrain.pth')
    parser.add_argument('-pseudo_path', action='store', dest='pseudo_path',
                        help='path to pseudo saliency map generated by Deepgaze II on COCO', default=None)
    args = parser.parse_args()

    if not exists(join(args.save_dir, args.model_name)):
        if not exists(args.save_dir):
            mkdir(args.save_dir)
            mkdir(join(args.save_dir, args.model_name))
        else:
            mkdir(join(args.save_dir, args.model_name))

    start_train(model_type=args.model_type, batch_size=args.batch_size, dataset_name=args.dataset_name,
                dataset_path=args.dataset_path, teacher_path=args.teacher_path, direct=args.save_dir,
                model_name=args.model_name, pretrain_path=args.pretrain_path, pseudo_path=args.pseudo_path)
    # python fine_tune.py -model_type A -batch_size 30 -dataset_name salicon -dataset_path /data/Datasets/SALICON/
    # -teacher_path /home/feiyan/data/Github/SAL_compress/salgan_pytorch/trained_models/salgan_adversarial2/models/
    # gen_42.pt -save_dir checkpoint -model_name finetune_salicon -pretrain_path weights/salicon_salgan_pretrain.pth

    '''python fine_tune.py -model_type C -batch_size 30 -dataset_name coco -dataset_path /data/coco/ 
    -teacher_path /home/feiyan/data/Github/SAL_compress/salgan_pytorch/trained_models/salgan_adversarial2/models/
    gen_42.pt -save_dir checkpoint -model_name finetune_coco -pretrain_path weights/coco_salgan_pretrain.pth 
    -pseudo_path /home/feiyan/data/generated_sal/deep_gaze_II_coco_with_cb_npy/ '''

