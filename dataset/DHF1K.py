import os
from torch.utils.data import Dataset
import numpy as np
from .utils import read_caffe_img, read_vgg_img, read_saliency, read_fixation

import matplotlib.pylab as plt

PATH_DHF1K = '/data/DHF1K/'
def gene_list_names(data, path_annotation):
    vid_ids = ['{:04d}'.format(i + 1) for i in data]
    n_frames = [n.split('.')[0] for vid_id in vid_ids
                     for n in os.listdir(path_annotation + '/' + vid_id) if '.png' in n]
    n_frames = [int(n) for n in n_frames]
    file_list = ['{}/{:04d}.png'.format(vid_id, n+1) for vid_id, n_frame in zip(vid_ids, n_frames)
                 for n in range(n_frame)]
    return np.asarray(file_list)
def gene_test_list(data, path_annotation):
    vid_ids = ['{:03d}'.format(i + 1) for i in data]
    frame_list = []
    for vid_id in vid_ids:
        print(vid_id)
        for n in os.listdir(path_annotation + '/' + vid_id):
            if '.png' in n:
                frame_list.append('{}/{}'.format(vid_id, n.replace('.png','')))
    frame_list.sort()
    return np.asarray(frame_list)
class dataset(Dataset):
    def __init__(self, data_path, mode='train', type=('caffe_img', 'vgg_img', 'sal_img'),
                 size=((192, 256), (192, 256), (192, 256)), N=None):
        self.size = size
        self.type = type
        self.mode = mode
        self.path_dataset = data_path
        self.path_images = os.path.join(self.path_dataset, 'frames')
        self.path_annotation = os.path.join(self.path_dataset, 'annotation')
        print(self.path_images,'!!!', self.size, '!!!')
        # get list images
        if mode == 'train':
            self.list_names = gene_list_names(range(600), self.path_annotation)
        elif mode == 'val':
            self.list_names = gene_list_names(range(600, 700), self.path_annotation)
            #if N:
        elif mode == 'test':
            self.list_names = gene_test_list(range(700, 1000), self.path_images)
        print(self.list_names[0])

        print("Init dataset in mode {}".format(mode))
        print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        #print(self.list_names[0])
        ima_name = self.list_names[index]
        if self.mode == 'test':
            img_path = os.path.join(self.path_images, ima_name+'.png')
        else:
            img_path = os.path.join(self.path_images, ima_name[1:])
        sal_name = ima_name.split('/')
        sal_path = os.path.join(self.path_annotation, sal_name[0]+'/maps/'+sal_name[1])
        pts_path = os.path.join(self.path_annotation, sal_name[0]+'/fixation/'+sal_name[1])
        all_dt = {}

        if 'caffe_img' in self.type:
            #print(img_path)
            cf_img = read_caffe_img(img_path, self.size[self.type.index('caffe_img')])
            all_dt['caffe_img'] = cf_img
        if 'vgg_img' in self.type:
            vgg_img = read_vgg_img(img_path, self.size[self.type.index('vgg_img')])
            all_dt['vgg_img'] = vgg_img
        if 'sal_img' in self.type:
            sal_img = read_saliency(sal_path, self.size[self.type.index('sal_img')])
            all_dt['sal_img'] = sal_img
        if 'fixation' in self.type:
            fixation = plt.imread(pts_path)
            all_dt['fixation'] = fixation
        if 'fixation_path' in self.type:
            all_dt['fixation_path'] = pts_path
        return all_dt

if __name__ == '__main__':
    d = dataset('train')
    print(d[0][0].shape, d[0][1][0].shape, d[0][2].shape)