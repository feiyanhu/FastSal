import glob
import os
import numpy as np
import cv2
from scipy.io import loadmat
import torch
from torchvision import transforms
from PIL import Image

PATH_SALICON_17 = "/data/Datasets/LSUN17"
PATH_SALICON_16 = "/media/eva/WD8T/Datasets/LSUN16"
PATH_IMAGES = "/media/eva/WD8T/Datasets/LSUN17/images"


def parse_fixations_17(path_file, shape=None):
    # read matlab
    X = loadmat(path_file)
    # create fixations
    if shape is None:
        shape = (X['resolution'][0][0], X['resolution'][0][1])

    fixations = np.zeros((shape[0], shape[1]))
    N = X['gaze']['fixations'].shape[0]
    # loop over all annotators
    fixations_coo = []
    count = 0
    for i in range(N):
        n_points = X['gaze']['fixations'][i][0].shape[0]
        # print(n_points)
        count+=n_points
        for k in range(n_points):
            x, y = X['gaze']['fixations'][i][0][k]
            #rescale the coordinates
            y = int((y/float(X['resolution'][0][0]))*shape[0])
            x = int((x/float(X['resolution'][0][1]))*shape[1])
            fixations_coo.append((x,y))
            fixations[y-1, x-1] = 1

    return fixations, np.array(fixations_coo)

def parse_fixations_16(path_file, shape=None):
    # read matlab
    X = loadmat(path_file)
    # create fixations
    if shape is None:
        shape = (X['resolution'][0][0], X['resolution'][0][1])
    fixations = np.zeros((shape[0], shape[1]))
    N = X['gaze']['fixations'][0].shape[0]
    fixations_coo = []
    count = 0
    # loop over all annotators
    for i in range(N):
        n_points = X['gaze']['fixations'][0][i].shape[0]
        # print(n_points)
        count+=n_points
        for k in range(n_points):
            x, y = X['gaze']['fixations'][0][i][k]
            #rescale the coordinates
            y = int((y/float(X['resolution'][0][0]))*shape[0])
            x = int((x/float(X['resolution'][0][1]))*shape[1])
            fixations_coo.append((x,y))
            fixations[y-1, x-1] = 1

    return fixations, np.array(fixations_coo)

def generate_shuffled_fixation(lsun, size=None):
    '''
    Generate aggregated random fixations for evaluation.

    args:   lsun version
            size of images
    '''

    if lsun==17:
        list_files = glob.glob(os.path.join(PATH_SALICON_17, 'train', '*.mat'))
        _parse_fixations = parse_fixations_17
    else:
        list_files = glob.glob(os.path.join(PATH_SALICON_16, 'train', '*.mat'))
        _parse_fixations = parse_fixations_16

    if size is None:
        im = cv2.imread(list_files[0].split('.')[0]+'.png', 0)
        size = im.shape

    # sample random 100 images for shuffle map
    shffledMap = None
    np.random.shuffle(list_files)
    for filename in list_files[:100]:
        sh, _ = _parse_fixations(filename, size)
        if shffledMap is None:
            shffledMap = sh
        else:
            shffledMap += sh
    shffledMap[shffledMap>0]=1

    return shffledMap

def parse_fixations(path_file, shape=None):
    # read matlab
    X = loadmat(path_file)
    # create fixations
    if shape is None:
        shape = (X['resolution'][0][0], X['resolution'][0][1])

    fixations = np.zeros((shape[0], shape[1]))
    N = X['gaze']['fixations'].shape[0]
    # loop over all annotators
    fixations_coo = []
    for i in range(N):
        n_points = X['gaze']['fixations'][i][0].shape[0]
        for k in range(n_points):
            x, y = X['gaze']['fixations'][i][0][k]

            #rescale the coordinates
            y = int((y/float(X['resolution'][0][0]))*shape[0])
            x = int((x/float(X['resolution'][0][1]))*shape[1])

            fixations_coo.append((x, y))
            fixations[y-1, x-1] = 1
    fixations_coo = np.array(fixations_coo)
    fixations_length = fixations_coo.shape[0]
    pad_matrix = np.zeros((480*640-fixations_coo.shape[0], 2))
    fixations_coo = np.concatenate((fixations_coo, pad_matrix), axis=0)
    #print(fixations_coo.shape, fixations_length)

    return fixations, fixations_coo, fixations_length

def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        #img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    #if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        #return img[np.newaxis]
    #else:
        # transpose (H, W, C) -> (C, H, W)
        #return img.transpose((2, 0, 1))
    return img


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    #img = torch.from_numpy(img).float()
    img = normalize(img)
    return img
    #return img.numpy()


def read_caffe_img(path, target_size, mean=[103.939, 116.779, 123.68]):
    image = cv2.imread(path)
    original_size = image.shape[0:2]
    if isinstance(target_size, tuple) or isinstance(target_size, list):
        if (target_size[0] != original_size[0] or target_size[1] != original_size[1]):
            image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image -= mean
    image = torch.FloatTensor(image)
    image = image.permute(2, 0, 1)
    return image


def read_vgg_img(path, target_size):
    vgg_img = read_image(path, color=True)
    original_size = vgg_img.size
    if isinstance(target_size, tuple) or isinstance(target_size, list):
        if (target_size[0] != original_size[1] or target_size[1] != original_size[0]):
            vgg_img = vgg_img.resize((target_size[1], target_size[0]), Image.ANTIALIAS)
        elif isinstance(target_size, int):
            vgg_img = vgg_img.resize((int(original_size[0]/target_size), int(original_size[2]/target_size))
                                     , Image.ANTIALIAS)
    vgg_img = np.asarray(vgg_img, dtype=np.float32)
    #print(vgg_img.shape,'???')
    vgg_img = pytorch_normalze(torch.FloatTensor(vgg_img).permute(2, 0, 1) / 255.0)
    return vgg_img, np.asarray(original_size)


def read_saliency(path, target_size):
    saliency = cv2.imread(path, 0)
    original_size = saliency.shape[0:2]
    if isinstance(target_size, tuple) or isinstance(target_size, list):
        if (target_size[0] != original_size[0] or target_size[1] != original_size[1]):
            saliency = cv2.resize(saliency, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    saliency = saliency.astype(np.float32)
    saliency = torch.FloatTensor(saliency)
    return saliency


def read_fixation(path, target_size):
    fixations, fixations_coo, fixations_length = parse_fixations(path, target_size)
    return fixations, fixations_coo, fixations_length

def resize_interpolate(npy_dt, size):
    npy_dt = torch.from_numpy(npy_dt[np.newaxis, np.newaxis, :, :])
    npy_dt = torch.exp(npy_dt)
    npy_dt = torch.nn.functional.interpolate(npy_dt, size=size, align_corners=True, mode='bicubic')
    npy_dt = npy_dt/torch.sum(npy_dt)
    #print(npy_dt.shape, t.sum(npy_dt))
    npy_dt = npy_dt[0, 0, :, :]
    return npy_dt

if __name__=='__main__':
    img = generate_shuffled_fixation(17)
    print(img.shape)