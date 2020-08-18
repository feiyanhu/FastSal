import numpy as np
import torch as t
import cv2
from torch.utils.data import DataLoader
import argparse

from model import student_teacher
from metrics.metrics import (AUC_Judd, AUC_Borji, AUC_shuffled, NSS, CC)
from metrics.utils import normalize_map
import dataset.dataset as dataset
import dataset.mit1003 as mit1003
from dataset.utils import generate_shuffled_fixation
from utils import load_weight


def evaluate(sal_pred, sal_gt, fixations_dt, shffledMap, original_shapes):
    sal_pred = sal_pred.detach().cpu().numpy()
    sal_gt = sal_gt.numpy()
    try:
        fixations_dt, fixations_coo_dt, fixations_length_dt = fixations_dt
        fixations_coo_dt = [x[0:fixations_length_dt[i], :].astype(np.int64) for i, x in
                            enumerate(fixations_coo_dt.numpy())]
    except:
        fixations_coo_dt = [None]*fixations_dt.shape[0]
    for fixations, fixations_coo, pred, gt, original_shape in \
            zip(fixations_dt, fixations_coo_dt, sal_pred, sal_gt, original_shapes):
        #print(fixations_coo, fixations_coo.shape)
        if fixations_coo is not None:
            others = np.copy(shffledMap)
            for x, y in fixations_coo:
                others[y - 1][x - 1] = 0
        pred = cv2.resize(pred[0], tuple(original_shape), interpolation=cv2.INTER_AREA)
        pred = normalize_map(pred)
        pred = cv2.GaussianBlur(pred, (5, 5), 0)
        pred = np.clip(pred, 0, 1)

        auc_judd_score = AUC_Judd(pred, fixations)
        auc_borji = AUC_Borji(pred, fixations)
        cc = CC(pred, gt)
        nss = NSS(pred, fixations)
        if fixations_coo is not None:
            auc_shuffled = AUC_shuffled(pred, fixations, others)
            yield auc_judd_score, auc_shuffled, auc_borji, nss, cc
        else:
            yield auc_judd_score, auc_borji, nss, cc

def train_one(model, dataloader, mode):
    try: shffledMap = np.load('dataset/salicon_shuffledMap.npy')
    except:
        shffledMap = generate_shuffled_fixation(17)
        np.save('dataset/salicon_shuffledMap.npy', shffledMap)
    all_metrics = []
    for i, X in enumerate(dataloader[mode]):
        fixations_path = X['fixation']
        vgg_inputs = X['vgg_img'][0].cuda()
        original_shape = X['vgg_img'][1].numpy()
        gt_maps_input = X['sal_img']
        sal_pred = model.predict(vgg_inputs)
        for mm in evaluate(sal_pred, gt_maps_input, fixations_path, shffledMap, original_shape):
            all_metrics.append(mm)
        if i%10 == 0:
            print('{} current accumulated loss '.format(i), np.mean(all_metrics, axis=0))
    all_metrics = np.mean(all_metrics, axis=0)
    new_metric = {}
    if all_metrics.shape[0] == 4:
        metric_name = ['aucj', 'aucb', 'nss', 'cc']
    elif all_metrics.shape[0] == 5:
        metric_name = ['aucj', 'aucs', 'aucb', 'nss', 'cc']
    for i, x in enumerate(all_metrics):new_metric[metric_name[i]] = x
    print(new_metric)

def start_eval(model_type, batch_size, dataset_name, dataset_path, model_path=None):
    if dataset_name == 'salicon':
        ds_validate = dataset.Salicon(dataset_path, mode='val', type=['vgg_img', 'sal_img', 'fixation'],
                                      size=[(192, 256), (480, 640), (480, 640)])
    elif dataset_name == 'mit1003':
        ds_validate = mit1003.dataset(dataset_path, type=['vgg_img', 'sal_img', 'fixation'],
                                      size=[(192, 256), None, None])

    if model_path:
        state_dict, opt_state = load_weight(model_path, remove_decoder=False)
    else:
        print('please specify trained models.')
        exit()

    model = student_teacher.salgan_teacher_student(False, model_type)
    model.student_net.load_state_dict(state_dict)
    model.cuda()


    # model.generator.load_state_dict(state_dict['state_dict'])
    # Dataloaders
    dataloader = {
        'val': DataLoader(ds_validate, batch_size=batch_size,
                          shuffle=False, num_workers=4)
    }

    model.eval()
    with t.no_grad():
        train_one(model, dataloader, 'val')

if __name__ == '__main__':
    coco_c = 'weights/coco_C.pth'  # coco_C
    coco_a = 'weights/coco_A.pth'  # coco_A
    salicon_c = 'weights/salicon_C.pth'  # salicon_C
    salicon_a = 'weights/salicon_A.pth'  # coco_A

    salicon_path = '/data/Datasets/SALICON/'
    mit1003_path = '/data/saliency_datasets/MIT1003_bak/'

    parser = argparse.ArgumentParser(description='configs for pretrain.')
    parser.add_argument('-model_type', action='store', dest='model_type',
                        help='model type can be either C(oncatenation) or A(ddition)', default='A')
    parser.add_argument('-batch_size', action='store', dest='batch_size',
                        help='batch size', default=10, type=int)
    parser.add_argument('-dataset_name', action='store', dest='dataset_name',
                        help='dataset_name either coco or salicon', default='salicon')
    parser.add_argument('-dataset_path', action='store', dest='dataset_path',
                        help='path to dataset')
    parser.add_argument('-model_path', action='store', dest='model_path',
                        help='path to trained FastSal model weights.')
    args = parser.parse_args()

    start_eval(model_type=args.model_type, batch_size=args.batch_size, dataset_name=args.dataset_name,
               dataset_path=args.dataset_path, model_path=args.model_path)
