import caffe
import numpy as np
import math
from PIL import Image
import scipy.io
import random

def Affinemat(angle, sx, sy, x0, y0, center = None, new_center = None):
    #angle = -angle/180.0*math.pi
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    if center is None:
        x = 0
        y = 0
    else:
        x = center[0]
        y = center[1]
    if new_center is None:
        nx = 0
        ny = 0
    else:
        nx = new_center[0]
        ny = new_center[1]
    a = cosine / sx
    b = sine / sx
    c = x-nx*a-ny*b
    d = -sine / sy
    e = cosine /sy
    f = y-nx*d-ny*e
    return (a,b,c,d,e,f)

def hard2softcha(hard_msk, cls):
    # hard_msk: num*cha*h*w
    if hard_msk.shape[1] > 1:
        raise Exception("hard2softcha takes single channel image/matrix only.")
    shape = hard_msk.shape
    soft_msk = np.zeros((shape[0],cls,shape[2],shape[3]))
    for ib in range(cls):
        tmp_msk = np.zeros((shape[0],1,shape[2],shape[3]))
        idx = np.nonzero(hard_msk==ib)
        tmp_msk[idx] = 1
        soft_msk[:,ib][:,np.newaxis] = tmp_msk
    return soft_msk

def crop_all(image, label, crop_dims):
    image = image[np.newaxis,:]
    label = label[np.newaxis,:]
    images = caffe.io.oversample(image,crop_dims)
    labels = caffe.io.oversample(label,crop_dims)
    images = images.transpose((1,2,3,0))
    labels = labels.transpose((1,2,3,0))
    images = np.array(images, dtype=np.float32)
    images = images.transpose((3,2,0,1))
    labels = np.array(labels, dtype=np.uint8)
    labels = labels.transpose((3,2,0,1))
    return images, labels

def crop_random(image, label, crop_dims):
    images,labels = randsample(image,label,crop_dims)
    images = np.array(images, dtype=np.float32)
    images = images.transpose((3,2,0,1))
    labels = np.array(labels, dtype=np.uint8)
    labels = labels.transpose((3,2,0,1))
    return images, labels

def randsample(image,label,crop_dims):
    # image: h*w*k
    # images: h*w*k*n
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims,dtype=int)
    images = np.zeros((crop_dims[0],crop_dims[1],im_shape[2],10))
    labels = np.zeros((crop_dims[0],crop_dims[1],1,10))
    count = 0
    protect = 0
    if len(np.unique(label[label<21])) < 2:
        mmp = 0
    else:
        mmp = 1

    while count<10:
        top = np.random.rand(1)* (im_shape[0]-crop_dims[0])
        left = np.random.rand(1)* (im_shape[1]-crop_dims[1])
        top = int(top)
        left = int(left)
        tmp_lb = label[top:top+crop_dims[0],left:left+crop_dims[1],:]
        nidx = np.unique(tmp_lb[tmp_lb<21])
        if (len(nidx) < 2 and mmp) and protect < 10:
            protect += 1
            continue
        protect = 0
        images[:,:,:,count] = image[top:top+crop_dims[0],left:left+crop_dims[1],:]
        labels[:,:,:,count] = tmp_lb
        count += 1
    return images,labels

def crop_random_v2(image, label, hard, crop_dims):
    images,labels = randsample_v2(image,label,hard,crop_dims)
    images = np.array(images, dtype=np.float32)
    images = images.transpose((3,2,0,1))
    labels = np.array(labels, dtype=np.uint8)
    labels = labels.transpose((3,2,0,1))
    return images, labels

def crop_binary(image, label, crop_dims):
    images,labels = randbinary_v2(image,label,crop_dims)
    images = np.array(images, dtype=np.float32)
    images = images.transpose((3,2,0,1))
    labels = np.array(labels, dtype=np.uint8)
    labels = labels.transpose((3,2,0,1))
    return images, labels

def randsample_v2(image,label,hard,crop_dims):
    # image: h*w*k
    # images: h*w*k*n
    # sample according to the pretrained model
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims,dtype=int)
    images = np.zeros((crop_dims[0],crop_dims[1],im_shape[2],10))
    labels = np.zeros((crop_dims[0],crop_dims[1],1,10))
    count = 0
    protect = 0
    if len(np.unique(hard[hard<21])) < 2:
        mmp = 0
    else:
        mmp = 1

    while count<10:
        top = np.random.rand(1)* (im_shape[0]-crop_dims[0])
        left = np.random.rand(1)* (im_shape[1]-crop_dims[1])
        top = int(top)
        left = int(left)
        tmp_hard = hard[top:top+crop_dims[0],left:left+crop_dims[1]]
        nidx = np.unique(tmp_hard[tmp_hard<21])
        if (len(nidx) < 2 and mmp) and protect < 10:
            protect += 1
            continue
        protect = 0
        images[:,:,:,count] = image[top:top+crop_dims[0],left:left+crop_dims[1],:]
        labels[:,:,:,count] = label[top:top+crop_dims[0],left:left+crop_dims[1],:]
        count += 1
    return images,labels

def randbinary(image,label,crop_dims):
    # image: h*w*k
    # images: h*w*k*n
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims,dtype=int)
    images = np.zeros((crop_dims[0],crop_dims[1],4,10))
    labels = np.zeros((crop_dims[0],crop_dims[1],1,10))
    count = 0
    protect = 0
    if len(np.unique(label[label<21])) < 2:
        mmp = 0
    else:
        mmp = 1

    while count<10:
        top = np.random.rand(1)* (im_shape[0]-crop_dims[0])
        left = np.random.rand(1)* (im_shape[1]-crop_dims[1])
        top = int(top)
        left = int(left)
        tmp_lb = label[top:top+crop_dims[0],left:left+crop_dims[1],:]
        nidx = np.unique(tmp_lb[tmp_lb<22])
        valid = len(nidx)
        if valid == 0:
            scipy.io.savemat('problem.mat',dict(image = image, label=label))
            continue
        if (valid == 1 and mmp) and protect < 10:
            protect += 1
            continue
        protect = 0

        images[:,:,0:3,count] = image[top:top+crop_dims[0],left:left+crop_dims[1],0:3]
        tmp_bi_lb = np.zeros_like(tmp_lb,dtype=np.float32)

        if valid == 1:
            # print image[top:top+crop_dims[0],left:left+crop_dims[1],nidx+3].shape, nidx
            images[:,:,3,count] = image[top:top+crop_dims[0],left:left+crop_dims[1],nidx[0]+3]
            tmp_bi_lb[tmp_lb==nidx[0]]=1
        else:
            # print valid, np.unique(tmp_lb)
            ind = random.randint(1,valid-1)
            # print image[top:top+crop_dims[0],left:left+crop_dims[1],nidx[ind]+3].shape,nidx[ind],ind
            images[:,:,3,count] = image[top:top+crop_dims[0],left:left+crop_dims[1],nidx[ind]+3]
            tmp_bi_lb[tmp_lb==nidx[ind]]=1
        labels[:,:,:,count] = tmp_bi_lb
        count += 1
    return images,labels


def randbinary_v2(image,label,crop_dims):
    # image: h*w*k
    # images: h*w*k*n
    im_shape = np.array(image.shape)
    crop_dims = np.array(crop_dims,dtype=int)
    images = np.zeros((crop_dims[0],crop_dims[1],im_shape[2],10))
    labels = np.zeros((crop_dims[0],crop_dims[1],1,10))
    count = 0
    protect = 0
    pri_lab = np.zeros_like(label,dtype=int)
    pri_lab[image[:,:,-1]>0.4] = 1

    while count<10:
        top = np.random.rand(1) * (im_shape[0]-crop_dims[0])
        left = np.random.rand(1) * (im_shape[1]-crop_dims[1])
        top = int(top)
        left = int(left)
        # tmp_lb = label[top:top+crop_dims[0],left:left+crop_dims[1],:]
        tmp_pri_lab = pri_lab[top:top+crop_dims[0],left:left+crop_dims[1],:]
        nidx = np.unique(tmp_pri_lab)
        if (len(nidx) < 2 ) and protect < 10:
            protect += 1
            continue
        protect = 0
        images[:,:,:,count] = image[top:top+crop_dims[0],left:left+crop_dims[1],:]
        labels[:,:,:,count] = label[top:top+crop_dims[0],left:left+crop_dims[1],:]
        count += 1
    return images,labels