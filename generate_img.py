import numpy as np
import torch as t
from torch.utils.data import DataLoader
import cv2
from os.path import exists
from os import mkdir

from model import student_teacher
import dataset.dataset as dataset
import dataset.mit1003 as mit1003
import dataset.mit300 as mit300
import dataset.DHF1K as dhf1k
from metrics.utils import normalize_map
from utils import load_weight

def post_process_png(prediction, original_shape):
    prediction = normalize_map(prediction)
    prediction = (prediction * 255).astype(np.uint8)
    prediction = cv2.resize(prediction, (original_shape[0], original_shape[1]),
                            interpolation=cv2.INTER_CUBIC)
    prediction = cv2.GaussianBlur(prediction, (5, 5), 0)
    #prediction = normalize_map(prediction)
    prediction = np.clip(prediction, 0, 255)
    return prediction

def post_process_probability2(prediction, original_shape):
    #prediction = prediction[np.newaxis, np.newaxis, :,:]
    np.seterr(all='raise')
    (h,w) = prediction.shape
    prediction = t.from_numpy(prediction)
    prediction = prediction.view(1, -1)
    prediction = t.nn.functional.softmax(prediction,dim=1)
    prediction = prediction.unsqueeze(1)
    prediction = prediction.view(1,1,h,w)
    prediction = t.nn.functional.interpolate(prediction, size=(original_shape[1], original_shape[0]), mode='bicubic',
                                             align_corners=True)
    prediction[prediction <= 0] = 1e-12
    prediction = prediction/t.sum(prediction)
    prediction = prediction[0,0,:,:]
    return prediction


def train_one(model, dataloader, file_list, mode, save_path, probability_output):
    count = 0
    for i, X in enumerate(dataloader[mode]):
        #vgg_inputs = X[0].cuda()
        vgg_inputs = X['vgg_img'][0].cuda()
        original_shape = X['vgg_img'][1].numpy()
        sal_pred = model.predict(vgg_inputs)
        sal_pred = sal_pred.detach().cpu().numpy()
        #print(sal_pred.shape)
        #continue
        for j, prediction in enumerate(sal_pred[:,0,:,:]):
            if probability_output:
                prediction = post_process_probability2(prediction, original_shape[j])
                np.save(save_path + file_list[count] + '.npy', prediction)
            else:
                prediction = post_process_png(prediction, original_shape[j])
                if '/' in file_list[count]:
                    tmp_dir = '{}{}'.format(save_path, file_list[count].split('/')[0])
                    if not exists(tmp_dir):
                        mkdir(tmp_dir)
                cv2.imwrite(save_path + file_list[count] + '.png', prediction)
            print(prediction.shape, file_list[count], save_path)
            count += 1

def main(model_type, batch_size, dataset_name, dataset_path, size, width_bigger,
         pretrain_path, save_path, probability_output):
    # Datasets for SALICON
    if dataset_name == 'salicon':
        ds_validate = dataset.Salicon(dataset_path, mode='test', type=['vgg_img'],size=(size,))
    elif dataset_name == 'mit300':
        ds_validate = mit300.dataset(dataset_path, type=('vgg_img'), size=(size,))
        ds_validate.renew_list(width_bigger=width_bigger)
    elif dataset_name == 'mit1003':
        ds_validate = mit1003.dataset(dataset_path, type=('vgg_img'), size=(size,))
        ds_validate.renew_list(width_bigger=width_bigger)
    elif dataset_name == 'dhf1k':
        ds_validate = dhf1k.dataset(dataset_path, mode='test', type=('vgg_img'), size=(size,))

    file_list = ds_validate.list_names

    if pretrain_path:
        state_dict, opt_state = load_weight(pretrain_path, remove_decoder=False)
    else:
        print('please specify trained models.')
        exit()

    model = student_teacher.salgan_teacher_student(False, model_type, use_probability_gt=probability_output)
    model.student_net.load_state_dict(state_dict)
    model.cuda()
    dataloader = {
        'val': DataLoader(ds_validate, batch_size=batch_size,
                          shuffle=False, num_workers=4)
    }

    print('--------------------------------------------->>>>>>')
    model.eval()
    with t.no_grad():
        train_one(model, dataloader, file_list, 'val', save_path, probability_output)
    print('--------------------------------------------->>>>>>')
    #print('loss val {}'.format(loss_val))

if __name__ == '__main__':
    coco_c = 'weights/coco_C.pth'  # coco_C
    coco_a = 'weights/coco_A.pth'  # coco_A
    salicon_c = 'weights/salicon_C.pth'  # salicon_C
    salicon_a = 'weights/salicon_A.pth'  # coco_A
    dhf1k_a = '../DHF1K_model10/ft_5_0.07962239047258239.pth'

    coco_path = '/data/coco/'
    salicon_path = '/data/Datasets/SALICON/'
    mit1003_path = '/data/saliency_datasets/MIT1003_bak/'
    mit300_path = '/data/Datasets/MIT300/'
    dhf1k_path = '/data/DHF1K/'

    save_path = 'generated/'
    save_path = '/home/data/generated_dhf1k_model10/'

    h = 192; w = 256
    #h = 256; w = 192
    main('A', 10, 'dhf1k', dhf1k_path, (h, w), True, dhf1k_a, save_path, False)
    #main('C', 10, 'mit1003', mit1003_path, (h,w), True, coco_c, save_path, False)
    #main('C', 10, 'mit1003', mit1003_path, (w,h), False, coco_c, save_path, False)
