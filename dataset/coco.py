import os
from torch.utils.data import Dataset,ConcatDataset
import numpy as np
import matplotlib.pylab as plt
from .utils import read_caffe_img, read_vgg_img, read_saliency, resize_interpolate

PATH_COCO = '/data/coco/'

class COCO(Dataset):
    def __init__(self, data_path, mode='train', year='2017', type=('caffe_img', 'vgg_img'),
                 size=((192, 256)), N=None):
        self.size = size
        self.type = type
        #self.target_size = target_size
        #self.mean = [103.939, 116.779, 123.68]
        self.path_dataset = data_path
        self.pseudo_gt_path = data_path
        self.path_images = os.path.join(self.path_dataset, mode+year)
        print(self.path_images,'!!!')

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]

        print("Init dataset in mode {}".format(mode))
        print("\t total of {} images.".format(list_names.shape[0]))

    def set_pseudo_gt_path(self, pseudo_path):
        self.pseudo_gt_path = pseudo_path

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        ima_name = self.list_names[index] + '.jpg'
        img_path = os.path.join(self.path_images, ima_name)
        all_dt = {}

        if 'deepgaze2_img' in self.type:
            img_name = self.list_names[index] + '.png'
            deep_gaze_img = os.path.join(self.pseudo_gt_path, img_name)
            deep_gaze_pred = read_saliency(deep_gaze_img, self.size)
            all_dt['deepgaze2_img'] = deep_gaze_pred
        if 'npy_img' in self.type:
            img_name = self.list_names[index] + '.npy'
            npy_img = os.path.join(self.pseudo_gt_path, img_name)
            npy_dt = np.load(npy_img)
            npy_dt = resize_interpolate(npy_dt, self.size)
            all_dt['npy_img'] = npy_dt
        if 'caffe_img' in self.type:
            cf_img = read_caffe_img(img_path, self.size)
            all_dt['caffe_img'] = cf_img
        if 'vgg_img' in self.type:
            vgg_img = read_vgg_img(img_path, self.size)
            all_dt['vgg_img'] = vgg_img
        if 'scipy_img' in self.type:
            scipy_img = plt.imread(img_path)
            all_dt['scipy_img'] = scipy_img
        return all_dt


if __name__ == '__main__':
    a = COCO(mode='train', type='npy_img')
    print(a[0][0].shape)
    exit()
    print('!!!')
    c = COCO()
    d = COCO(mode='val')
    e = COCO(mode='test')
    f = ConcatDataset([c,d,e])
    print(len(f),'!!!', type(f))
    (a, b) = c[379]
    print(a.shape, b.shape)