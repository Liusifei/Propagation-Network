from __future__ import division
import caffe
import numpy as np
import os
import sys
import scipy.io
from datetime import datetime
from PIL import Image

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def image_padding(im_array, pad_dim):
    im_pad = np.zeros((pad_dim[0],pad_dim[1],im_array.shape[2]))
    # msk = np.zeros((pad_dim[0],pad_dim[1]))
    if im_array.shape[0] > pad_dim[0] or im_array.shape[1] > pad_dim[1]:
        for ch in range(im_array.shape[2]):
            im_array_tmp = np.array(im_array[:,:,ch],dtype=np.uint8)
            im_array_tmp = Image.fromarray(im_array_tmp)
            im_array_tmp = im_array_tmp.resize((pad_dim[0],pad_dim[1]), resample=Image.BILINEAR)
            im_array_tmp = np.array(im_array_tmp,dtype=np.uint8)
            im_pad[:,:,ch] = im_array_tmp
            # msk[...] = 1
            bound_x=0
            bound_y=0
    else:
        bound_y = int((pad_dim[0]-im_array.shape[0])/2)
        bound_x = int((pad_dim[1]-im_array.shape[1])/2)
        # msk[bound_y+1:bound_y+im_array.shape[0], bound_x+1:bound_x+im_array.shape[1]] = 1
        # bound_y = 0
        # bound_x = 0
        for ch in range(im_array.shape[2]):
            im_pad[bound_y:bound_y+im_array.shape[0], bound_x:bound_x+im_array.shape[1],:] = \
                im_array
    return im_pad, (bound_y,bound_x)

def im_from_padding(im_pad, im_dim, bound):
    if im_dim[0] > im_pad.shape[0] or im_dim[1] > im_pad.shape[1]:
        im_crop = np.zeros((im_dim[0],im_dim[1],im_pad.shape[2]))
        for ch in range(im_pad.shape[2]):
            im_pad_tmp = np.array(im_pad[:,:,ch],dtype=np.uint8)
            im_pad_tmp = Image.fromarray(im_pad_tmp)
            im_pad_tmp = im_pad_tmp.resize((im_dim[1],im_dim[0]),resample=Image.BILINEAR)
            im_crop[:,:,ch] = np.array(im_pad_tmp,dtype=np.uint8)
    else:
        im_crop = im_pad[bound[0]:bound[0]+im_dim[0], bound[1]:bound[1]+im_dim[1],:]
    # print im_crop.shape, im_pad.shape
    return im_crop

def compute_hist(net, dataroot, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir and not(os.path.exists(save_dir)):
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    # loss = 0
    for idx in dataset:
        print idx
        im = Image.open('{}/JPEGImages/{}.jpg'.format(dataroot, idx))
        lb = Image.open('{}/SegmentationClass/{}.png'.format(dataroot, idx))
        im_dim = (im.height,im.width)
        lb = np.array(lb,dtype=np.uint8)
        lb = lb[:,:,np.newaxis]

        # hard_rf = scipy.io.loadmat('data/pascal/VOCdevkit/VOC2012/SegmentationCNN/{}.mat'.format(idx))
        # hard_rf = np.array(hard_rf['out'], dtype=np.uint8)
        # valid_lbs = np.unique(hard_rf[hard_rf<22])
           

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_, bound = image_padding(in_, (512,512))

        image = in_.transpose((2,0,1))
        trans_lb_ = lb.transpose((2,0,1))

        net.blobs['data'].reshape(1, *image.shape)
        net.blobs['data'].data[...] = image
        net.forward()
        print layer
        prob = net.blobs[layer].data[0]
        prob_ = prob.transpose((1,2,0))
        prob_ = im_from_padding(prob_, im_dim, bound)
        prob_ = prob_.transpose((2,0,1))

        # prob_rect = np.zeros_like(prob_)
        # prob_rect[valid_lbs] = prob_[valid_lbs]       
        # prob_hard = prob_rect.argmax(0)
        prob_hard = prob_.argmax(0)

        hist += fast_hist(trans_lb_.flatten(),
                                prob_hard.flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(prob_hard.astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
            scipy.io.savemat(os.path.join(save_dir, idx + '.mat'), dict(prob = prob_))
    return hist


def seg_tests(net, iter, dataroot, save_format, dataset, layer='score', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist = compute_hist(net, dataroot, save_format, dataset, layer, gt)
    # mean loss
    # print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    return hist