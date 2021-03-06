import caffe
import numpy as np
import math
from PIL import Image
import scipy.io
import random
import sys
sys.path.append("..")
from pyutils import data_roidb as db_roi
# from multiprocessing import Process, Queue


class ROISampleDataLayer(caffe.Layer):
	"""
	Load (input image, cnn output, label image) pairs from PASCAL VOC
	one-at-a-time while reshaping the net to preserve dimensions.

	Use this to feed data to a rnn refine network with multiple label at once.
	"""

	def setup(self, bottom, top):
		"""
		roi_dim: roi width/height, usually set as 128; need to copy to roi layer param
		"""
		# config
		params = eval(self.param_str)
		self.USE_PREFETCH = params.get('USE_PREFETCH',False)
		self.root_dir = params['root']
		self.split = params['split']
		self.mean = np.array(params['mean'])
		self.random = params.get('randomize', True)
		self.seed = params.get('seed', None)
		self.shape = np.array(params['shape'])
		self.imagenorm = params.get('norm', False)
		self.roi_num = params['roi_num_per_im']
		roi_dim = params['roi_dim']
		if isinstance(roi_dim, int):
			self.roi_dim = (roi_dim,roi_dim)
		else:
			self.roi_dim = roi_dim

		if len(top) != 3:
			raise Exception("Need to define three tops: data, label and rois.")
		if len(bottom) != 0:
			raise Exception("Do not define a bottom.")

		# load indices for images and labels
		split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.root_dir,
				self.split)
		self.indices = open(split_f, 'r').read().splitlines()
		self.idx = np.array(range(self.shape[0]))

		random.seed(self.seed)
		for id in range(len(self.idx)):
			self.idx[id] = random.randint(0, len(self.indices)-1)


	def reshape(self, bottom, top):
		self.data = np.zeros(self.shape)
		self.label = np.zeros((self.shape[0],1,self.shape[2],self.shape[3]))
		self.rois = np.zeros((self.roi_num * self.shape[0], 5))

		# reshape tops to fit (leading 1 is for batch dimension)
		top[0].reshape(*self.data.shape)
		top[1].reshape(*self.label.shape)
		top[2].reshape(*self.rois.shape)

	def get_next_minibatch_idx(self):
		for id in range(len(self.idx)):
			self.idx[id] = random.randint(0, len(self.indices)-1)

	def get_next_minibatch(self):
		"""return everything"""
		self.get_next_minibatch_idx()
		for id in range(len(self.idx)):
			self.data[id], self.label[id]= \
				db_roi.load_imagelabel_ac(self.root_dir, self.indices[self.idx[id]], self.shape, self.mean)
			# print id, self.label[id].shape, self.roi_num, self.roi_dim
			self.rois[id*self.roi_num : (id+1)*self.roi_num] = \
				db_roi.getrois(id, self.label[id], self.roi_num, self.roi_dim)
			# print self.rois[id*self.roi_num : (id+1)*self.roi_num]

	
	def forward(self, bottom, top):
		self.get_next_minibatch()
		top[0].data[...] = self.data
		top[1].data[...] = self.label
		top[2].data[...] = self.rois

	
	def backward(self, top, propagate_down, bottom):
		pass