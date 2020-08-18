import os
from torch.utils.data import Dataset
import numpy as np
from .utils import read_caffe_img, read_vgg_img

import matplotlib.pylab as plt
import PIL.Image as pil_image

PATH_MIT300 = '/data/Datasets/MIT300/'

class dataset(Dataset):
    def __init__(self, data_path, type=('caffe_img', 'vgg_img'),
                 size=((192, 256),), N=None):
        global PATH_MIT300
        self.size = size
        self.type = type
        #self.target_size = target_size
        #self.mean = [103.939, 116.779, 123.68]
        self.path_dataset = data_path
        self.path_images = os.path.join(self.path_dataset, 'BenchmarkIMAGES')
        print(self.path_images,'!!!', self.size, '!!!')

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names if '.jpg' in n])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]

        #print("Init dataset in mode {}".format(mode))
        print("\t total of {} images.".format(list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        ima_name = self.list_names[index] + '.jpg'
        img_path = os.path.join(self.path_images, ima_name)
        all_dt = {}

        if 'caffe_img' in self.type:
            cf_img = read_caffe_img(img_path, self.size[self.type.index('caffe_img')])
            all_dt['caffe_img'] = cf_img
        if 'vgg_img' in self.type:
            #print('!!!!', self.type.index('vgg_img'), self.size)
            vgg_img = read_vgg_img(img_path, self.size[self.type.index('vgg_img')])
            #print(vgg_img[0].shape)
            all_dt['vgg_img'] = vgg_img
        if 'scipy_img' in self.type:
            scipy_img = plt.imread(img_path)
            all_dt['scipy_img'] = scipy_img
        return all_dt

    def renew_list(self, width_bigger=True):
        self.new_list_name = []
        for img_name in self.list_names:
            ima_name = img_name + '.jpg'
            img_path = os.path.join(self.path_images, ima_name)
            try:
                img_pil_open = pil_image.open(img_path, 'r')
            except:
                print(img_path)
                continue
            (W, H) = img_pil_open.size
            if width_bigger:
                if W >= H:
                    self.new_list_name.append(img_name)
            else:
                if H > W:
                    self.new_list_name.append(img_name)
        self.list_names = np.asarray(self.new_list_name)


if __name__ == '__main__':
    a = dataset(type=('vgg_img'), size=((192,256),))
    a.renew_list(width_bigger=False)
    print(len(a))
    print(a[0][0][0].shape,  a[0][0][1])
    #print(a[0][0], a[0][1])