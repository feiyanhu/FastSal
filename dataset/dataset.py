import os
import matplotlib.pylab as plt
from torch.utils.data import Dataset
import numpy as np
from .utils import read_caffe_img, read_vgg_img, read_saliency, read_fixation

# constants
#PATH_SALICON = "/media/eva/WD8T/Datasets/SALICON/"
PATH_SALICON = "/data/Datasets/SALICON"


class Salicon(Dataset):
    def __init__(self, dataset_path, mode='train', type=('caffe_img', 'vgg_img', 'sal_img', 'fixation'),
                 size=((192, 256), (192, 256), (192, 256), (192, 256)), N=None):
        self.size = size
        self.type = type
        self.path_dataset = dataset_path
        self.path_images = os.path.join(self.path_dataset, 'images')
        self.path_saliency = os.path.join(self.path_dataset, 'saliency')
        self.path_fixations = os.path.join(self.path_dataset, 'fixations', mode)

        # get list images
        list_names = os.listdir(self.path_fixations)
        list_names = np.array([n.split('.')[0] for n in list_names])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]

        print("Init dataset in mode {}".format(mode))
        print("\t total of {} images.".format(list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):

        # set path
        ima_name = self.list_names[index]+'.jpg'
        img_path = os.path.join(self.path_images, ima_name)
        ima_name = self.list_names[index]+'.png'
        sal_path = os.path.join(self.path_saliency, ima_name)
        fixations_path = os.path.join(self.path_fixations, self.list_names[index]+'.mat')
        all_dt = {}

        if 'caffe_img' in self.type:
            cf_img = read_caffe_img(img_path, self.size[self.type.index('caffe_img')])
            all_dt['caffe_img'] = cf_img
        if 'vgg_img' in self.type:
            vgg_img = read_vgg_img(img_path, self.size[self.type.index('vgg_img')])
            all_dt['vgg_img'] = vgg_img
        if 'sal_img' in self.type:
            sal_img = read_saliency(sal_path, self.size[self.type.index('sal_img')])
            all_dt['sal_img'] = sal_img/255.0
        if 'fixation' in self.type:
            fixation = read_fixation(fixations_path, self.size[self.type.index('fixation')])
            #print(len(fixation))
            all_dt['fixation'] = fixation
        if 'fixation_path' in self.type:
            all_dt['fixation_path'] = fixations_path
        if 'scipy_img' in self.type:
            scipy_img = plt.imread(img_path)
            all_dt['scipy_img'] = scipy_img
        #print(cf_img.shape, vgg_img.shape, sal_img.shape, fixation.shape)
        return all_dt


if __name__ == '__main__':
    ds = Salicon(mode='val')
    ds[100]
    exit()

    print(len(ds))
    for i in range(1, 6):
        image, saliency, fixations,_ = ds[i]

        plt.subplot(1,3,1)
        plt.imshow(np.transpose(image,(1,2,0)))
        plt.subplot(1,3,2)
        plt.imshow(saliency)
        plt.subplot(1,3,3)
        plt.imshow(fixations)
        print(fixations.shape, fixations[fixations>0])
        plt.show()
