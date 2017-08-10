import caffe
import numpy as np
import scipy.io
import math
import os
from PIL import Image
import time
caffe.set_mode_gpu()
caffe.set_device(0)
solverproto = 'roi_test.prototxt'
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('img.jpg')
im = im.resize((512,512),resample=Image.BILINEAR)
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_1 = in_.transpose((2,0,1))
in_2 = in_.transpose((2,1,0))
in_ = np.append(in_1[np.newaxis],in_2[np.newaxis],axis=0)

rois = np.array([[1,10,10,137,137],[1,17,27,144,154],[2,16,40,143,167],[2,40,26,167,153]])


# load net
net = caffe.Net(solverproto, caffe.TRAIN)
print 'finish init ...'
# shape for input (data blob is N x C x H x W), set data
# print in_.shape, prior.shape
net.blobs['image'].reshape(*in_.shape)
net.blobs['image'].data[...] = in_
print 'finish putting image data ...'
net.blobs['data_roi'].reshape(*rois.shape)
net.blobs['data_roi'].data[...] = rois
print 'finish putting roi data ...'


# run net and take argmax for prediction
net.forward()
print 'finish forward ...'

samples = net.blobs['samples'].data[...]
errors = 1-samples
net.blobs['samples'].diff[...] = errors
net.backward()
print 'finish backward ...'

diffs = net.blobs['image'].diff[...]
scipy.io.savemat('samples.mat', dict(samples = samples, diffs = diffs))
