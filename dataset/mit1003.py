import os
from torch.utils.data import Dataset
import numpy as np
from .utils import read_caffe_img, read_vgg_img, read_saliency

import matplotlib.pylab as plt
import PIL.Image as pil_image

PATH_MIT1003 = '/data/saliency_datasets/MIT1003_bak/'

class dataset(Dataset):
    def __init__(self, data_path, mode='train', type=('caffe_img', 'vgg_img'),
                 size=((192, 256))):
        global PATH_MIT1003
        self.size = size
        self.type = type
        #self.target_size = target_size
        #self.mean = [103.939, 116.779, 123.68]
        self.path_dataset = data_path
        self.path_images = os.path.join(self.path_dataset, 'ALLSTIMULI')
        #print(self.path_images,'!!!', self.size, '!!!')

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names if '.jpeg' in n])
        self.list_names = list_names

        print("Init dataset in mode {}".format(mode))
        print("\t total of {} images.".format(list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        ima_name = self.list_names[index] + '.jpeg'
        img_path = os.path.join(self.path_images, ima_name)
        sal_name = self.list_names[index] + '_fixMap.jpg'
        sal_path = os.path.join(self.path_images.replace('ALLSTIMULI', 'ALLFIXATIONMAPS'), sal_name)
        pts_name = self.list_names[index] + '_fixPts.jpg'
        pts_path = os.path.join(self.path_images.replace('ALLSTIMULI', 'ALLFIXATIONMAPS'), pts_name)
        all_dt = {}

        if 'sal_img' in self.type:
            sal = read_saliency(sal_path, self.size[self.type.index('sal_img')])
            all_dt['sal_img'] = sal/255.0
        if 'fixation' in self.type:
            pts = read_saliency(pts_path, self.size[self.type.index('fixation')])
            all_dt['fixation'] = pts/255.0
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
            ima_name = img_name + '.jpeg'
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
    a = Dataset(mode='train', type=('vgg_img', 'sal_img'), size=(None))
    print(a[0][0].shape, a[0][1][0].shape, a[0][1][1])