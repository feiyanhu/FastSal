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
    #print(batch_size,'!!!')
    ds_train = dataset.Salicon(dataset_path, mode='train')
    ds_validate = dataset.Salicon(dataset_path, mode='val')

    dataloader = {
        'train': DataLoader(ds_train, batch_size=batch_size,
                            shuffle=True, num_workers=4),
        'val': DataLoader(ds_validate, batch_size=int(batch_size/2),
                          shuffle=False, num_workers=4)
    }
    return dataloader

def coco_data(batch_size, dataset_path):
    #size = (96, 128)
    size = (192, 256)
    ds_train = coco.COCO(dataset_path, mode='train', size=size)  # N=100)
    ds_test = coco.COCO(dataset_path, mode='test', size=size)
    ds_train = ConcatDataset([ds_train, ds_test])
    ds_validate = coco.COCO(dataset_path, mode='val', size=size)
    dataloader = {
        'train': DataLoader(ds_train, batch_size=batch_size,
                            shuffle=True, num_workers=4),
        'val': DataLoader(ds_validate, batch_size=int(batch_size/4),
                          shuffle=False, num_workers=4),
    }
    return dataloader

def train_one(model, dataloader, optimizer, mode):
    all_loss = []
    for i, X in enumerate(dataloader[mode]):
        optimizer.zero_grad()
        inputs = X['caffe_img'].cuda()
        vgg_inputs = X['vgg_img'][0].cuda()
        losses = model.forward(inputs, vgg_inputs)
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

def start_train(batch_size, dataset_name, dataset_path, teacher_path, direct, model_name):
    if dataset_name == 'salicon':
        dataloader = salicon_data(batch_size, dataset_path)
    elif dataset_name == 'coco':
        dataloader = coco_data(batch_size, dataset_path)

    model = student_teacher.salgan_teacher_student(True, 'C', teacher_path)
    model.cuda()

    lr = 0.01
    lr_decay = 0.1
    optimizer = model.get_optimizer(lr)
    smallest_val = None
    best_epoch = None

    for epoch in range(0, 100, 1):
        model.train()
        loss_train, model = train_one(model, dataloader, optimizer, 'train')
        print('{} loss train {}, lr {}'.format(epoch, loss_train, lr))
        print('--------------------------------------------->>>>>>')
        model.eval()
        loss_val, model = train_one(model, dataloader, optimizer, 'val')
        print('--------------------------------------------->>>>>>')
        print('{} loss val {}'.format(epoch, loss_val))

        smallest_val, best_epoch, model, optimizer = save_weight(smallest_val, best_epoch, loss_val, epoch,
                                                                 direct, model_name, model, optimizer)
        if epoch == 15 or epoch == 30 or epoch == 60:
            path = '{}/{}/{}_{:f}.pth'.format(direct, model_name, best_epoch, smallest_val)
            state_dict, opt_state = load_weight(path, remove_decoder=False)
            model.student_net.load_state_dict(state_dict)
            # optimizer.load_state_dict(state_dict['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            lr = lr * lr_decay


if __name__ == '__main__':
    path = '/home/feiyan/data/Github/SAL_compress/salgan_pytorch/trained_models/' \
           'salgan_adversarial2/models/gen_42.pt'

    coco_path = '/data/coco/'
    salicon_path = '/data/Datasets/SALICON/'

    parser = argparse.ArgumentParser(description='configs for pretrain.')
    parser.add_argument('-batch_size', action='store', dest='batch_size',
                        help='batch size', default=10, type=int)
    parser.add_argument('-dataset_name', action='store', dest='dataset_name',
                        help='dataset_name either coco or salicon')
    parser.add_argument('-dataset_path', action='store', dest='dataset_path',
                        help='path to dataset')
    parser.add_argument('-teacher_path', action='store', dest='teacher_path',
                        help='path to teacher weight (SALGAN)')
    parser.add_argument('-save_dir', action='store', dest='save_dir',
                        help='directory to save pretrained weights', default='checkpoint')
    parser.add_argument('-model_name', action='store', dest='model_name',
                        help='pretrained model name', default='pretrained_model')
    args = parser.parse_args()

    if not exists(join(args.save_dir, args.model_name)):
        if not exists(args.save_dir):
            mkdir(args.save_dir)
            mkdir(join(args.save_dir, args.model_name))
        else:
            mkdir(join(args.save_dir, args.model_name))
    start_train(batch_size=args.batch_size, dataset_name=args.dataset_name, dataset_path=args.dataset_path,
                teacher_path=args.teacher_path, direct=args.save_dir, model_name=args.model_name)

