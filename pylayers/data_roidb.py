import caffe
import numpy as np
import math
from PIL import Image
import scipy.io
import random
sys.path.append("..")
from pyutils import datalayer_util as du

def load_imagelabel_ac(voc_dir, idx, shape, mean_):
	width_ = np.array(1.2*shape[3],dtype=int)
	height_ = np.array(1.2*shape[2],dtype=int)
	im = Image.open('{}/JPEGImages/{}.jpg'.format(voc_dir, idx))
	lb = Image.open('{}/SegmentationClass/{}.png'.format(voc_dir, idx))
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
	image = np.array(im,dtype=np.float32)
	image = image[:,:,::-1]
	image -= mean_
	label = np.array(lb,dtype=np.uint8)
	label = label[:,:,np.newaxis]

    image = image.transpose((2,0,1))
    label = label.transpose((2,0,1))
	  
	return image, label


def getrois(id_, label, roi_num, roi_dim):
	# image: h*w*k
	# images: h*w*k*n
	# sample according to the pretrained model
	im_shape = np.array(label.shape)
	count = 0
	protect = 0
	rois = np.zeros((roi_num,5))
	if len(np.unique(label[label<21])) < 2:
		mmp = 0
	else:
		mmp = 1

	while count < roi_num:
		top = int(np.random.rand(1)* (im_shape[0]-roi_dim[0]))
		left = int(np.random.rand(1)* (im_shape[1]-roi_dim[1]))
		tmp_lb = label[0,top:top+roi_dim[0],left:left+roi_dim[1]]
		nidx = np.unique(tmp_lb[tmp_lb<21])
		if (len(nidx) < 2 and mmp) and protect < 10:
			protect += 1
			continue
		protect = 0
		rois[count] = np.array((id_, left,top,roi_dim[0],roi_dim[1]))
		count += 1
	return rois