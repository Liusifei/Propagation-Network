import caffe
import numpy as np
import math
from PIL import Image
import scipy.io
import random
import sys
sys.path.append("..")
from pyutils import datalayer_util as du

class AgRcProbVOCDataLayer(caffe.Layer):
    """
    Load (input image, cnn output, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.

    Use this to feed data to a rnn refine network with multiple label at once.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL VOC semantic segmentation.

        example

        params = dict(voc_dir="/path/to/PASCAL/VOC2011",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.shape = np.array(params['shape'])
        self.refine = params.get('refine', True)
        self.sample_size = self.shape[0] / 10 #default 4
        self.crop_dims = (self.shape[2], self.shape[3])

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.voc_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = np.array(range(self.sample_size))

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            for id in range(len(self.idx)):
                self.idx[id] = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        self.data = np.zeros(self.shape)
        self.label = np.zeros((self.shape[0],1,self.shape[2],self.shape[3]))

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)


    def forward(self, bottom, top):
        for id in range(len(self.idx)):
            # load image + label image pair
            image, label, hard_rf= \
            self.load_imagelabel_ac(self.indices[self.idx[id]],self.shape)
            self.data[id*10:(id+1)*10], self.label[id*10:(id+1)*10] = \
            du.crop_random_v2(image, label, hard_rf, self.crop_dims)
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        # pick next input
        if self.random:
            for id in range(len(self.idx)):
                self.idx[id] = random.randint(0, len(self.indices)-1)
        else:
            self.idx += self.shape[0]
            if self.idx[-1] >= len(self.indices):
                self.idx = range(self.shape[0])


    def backward(self, top, propagate_down, bottom):
        pass


    def load_imagelabel_ac(self, idx, shape):
        width_ = np.array(1.2*shape[3],dtype=int)
        height_ = np.array(1.2*shape[2],dtype=int)
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
        lb = Image.open('{}/SegmentationClass/{}.png'.format(self.voc_dir, idx))
        if self.refine:
            hard_rf = scipy.io.loadmat('{}/SegmentationCNN/{}.mat'.format(self.voc_dir, idx))
            hard_rf = np.array(hard_rf['out'], dtype=np.uint8)
            rf = scipy.io.loadmat('{}/SegmentationProb/{}.mat'.format(self.voc_dir, idx))
            rf = np.array(rf['out'], dtype=np.float32)
            rf = rf.transpose((1,2,0))
        (x,y) = im.size
        center = (x/2,y/2)
        rate = (np.random.rand(1)-0.5)
        shift_x = np.maximum(x,y) * rate
        rate = (np.random.rand(1)-0.5)
        shift_y = np.maximum(x,y) * rate
        scale_x = 1+(np.random.rand(1)-0.5) /5.0
        scale_y = 1+(np.random.rand(1)-0.5) /5.0
        angle = (np.random.rand(1)-0.5)*(30.0/180.0)*math.pi
        mat = du.Affinemat(angle,scale_x,scale_y,center,(center[0]+shift_x,center[1]+shift_y))
        im = im.transform((x,y), Image.AFFINE, mat, resample=Image.BILINEAR)
        lb = lb.transform((x,y), Image.AFFINE, mat, resample=Image.BILINEAR)
        if x < width_ or y < height_:
            im = im.resize((width_,height_),resample=Image.BILINEAR)
            lb = lb.resize((width_,height_),resample=Image.BILINEAR)
        image = np.array(im,dtype=np.float32)
        image = image[:,:,::-1]
        image -= self.mean
        label = np.array(lb,dtype=np.uint8)
        label = label[:,:,np.newaxis]
        if self.refine:
            for ch in range(rf.shape[2]):
                if ch==0:
                    rf_tmp = 1-rf[:,:,ch]
                else:
                    rf_tmp = rf[:,:,ch]
                rf_tmp = np.array(rf_tmp * 255.0,dtype=np.uint8)
                rf_tmp = Image.fromarray(rf_tmp)
                rf_tmp = rf_tmp.transform((x,y), Image.AFFINE, mat, resample=Image.BILINEAR)
                if x < width_ or y < height_:
                    rf_tmp = rf_tmp.resize((width_,height_),resample=Image.BILINEAR)
                rf_tmp = np.array(rf_tmp,dtype=np.float32) / 255.0
                if ch==0:
                    rf_tmp = 1-rf_tmp
                    prior = np.zeros((rf_tmp.shape[0],rf_tmp.shape[1],rf.shape[2]))
                prior[:,:,ch] = rf_tmp
            image = np.append(image,prior,2)        
        return image, label, hard_rf