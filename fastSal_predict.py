import model.fastSal as fastsal
from dataset.utils import read_vgg_img
from utils import load_weight
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from os.path import isfile, isdir, join
from os import listdir
import numpy as np
import argparse
from generate_img import post_process_png, post_process_probability2
import cv2

class img_dataset(Dataset):
    def __init__(self, img_path, output_path):
        if isdir(img_path):
            print('image folder is {}'.format(img_path))
            file_list = [f for f in listdir(img_path) if isfile(join(img_path, f))]
            file_list = [f for f in file_list if '.jpg' in f or 'jpeg' in f or 'png' in f]
            self.file_list = np.asarray(file_list)
            self.dir = img_path
        elif isfile(img_path):
            print('image file is {}'.format(img_path))
            self.file_list = np.asarray([img_path])
            self.dir = None
        self.output_dir = output_path
    def __getitem__(self, item):
        if self.dir:
            img_path = join(self.dir, self.file_list[item])
            output_path = join(self.output_dir, 'out_' + self.file_list[item])
        else:
            img_path = self.file_list[item]
            output_path = self.output_dir
        vgg_img, original_size = read_vgg_img(img_path, (192, 256))
        return vgg_img, original_size, output_path
    def __len__(self):
        return self.file_list.shape[0]


def predict(model_type, finetune_dataset, input_path, output_path,
            probability_output, batch_size, gpu=True):
    model = fastsal.fastsal(pretrain_mode=False, model_type=model_type)
    state_dict, opt_state = load_weight('weights/{}_{}.pth'.format(finetune_dataset, model_type), remove_decoder=False)
    model.load_state_dict(state_dict)
    if gpu:
        model.cuda()
    simple_data = img_dataset(input_path, output_path)
    simple_loader = DataLoader(simple_data, batch_size=batch_size, shuffle=False, num_workers=4)
    for x, original_size_list, output_path_list in simple_loader:
        if gpu:
            x = x.float().cuda()
        y = model(x)
        if not probability_output: y = nn.Sigmoid()(y)
        if gpu:
            y = y.detach().cpu()
        y = y.numpy()
        for i, prediction in enumerate(y[:, 0, :, :]):
            img_output_path = output_path_list[i]
            original_size = original_size_list[i].numpy()
            print(img_output_path)
            if not probability_output:
                img_data = post_process_png(prediction, original_size)
                cv2.imwrite(img_output_path, img_data)
            else:
                img_data = post_process_probability2(prediction, original_size)
                np.save(img_output_path.split('.')[0], img_data)

if __name__ == '__main__':
    coco_c = 'weights/coco_C.pth'  # coco_C
    coco_a = 'weights/coco_A.pth'  # coco_A
    salicon_c = 'weights/salicon_C.pth'  # salicon_C
    salicon_a = 'weights/salicon_A.pth'  # coco_A

    parser = argparse.ArgumentParser(description='configs for predict.')
    parser.add_argument('-model_type', action='store', dest='model_type',
                        help='model type can be either C(oncatenation) or A(ddition)', default='A')
    parser.add_argument('-finetune_dataset', action='store', dest='finetune_dataset',
                        help='Dataset that the model fine tuned on.', default='salicon')
    parser.add_argument('-input_path', action='store', dest='input_path',
                        help='path to input image or image folder')
    parser.add_argument('-output_path', action='store', dest='output_path',
                        help='path to output image or image folder')
    parser.add_argument('-batch_size', action='store', dest='batch_size',
                        help='batch size.', default=2, type=int)
    parser.add_argument('-probability_output', action='store', dest='probability_output',
                        help='use probability_output or not', default=False, type=bool)
    parser.add_argument('-gpu', action='store', dest='gpu',
                        help='use gpu or not', default=True, type=bool)
    args = parser.parse_args()

    predict(args.model_type, args.finetune_dataset, args.input_path, args.output_path,
            args.probability_output, args.batch_size, gpu=args.gpu)
    #x = torch.zeros((10, 3, 192, 256))
    #y = model(x)
    #print(y.shape)