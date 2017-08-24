import caffe
import numpy as np
import math
from PIL import Image
import scipy.io
import random
import sys
from multiprocessing import Process, Queue
sys.path.append("..")
from pyutils import data_roidb as db_roi

class ROIsamplePrefatchLayer(caffe.Layer):
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
		self.roi_dim = params['roi_dim']

		self.sample_size = self.shape[0] / 10

		if len(top) != 3:
			raise Exception("Need to define three tops: data, label and rois.")
		if len(bottom) != 0:
			raise Exception("Do not define a bottom.")

		# load indices for images and labels
		split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.voc_dir,
				self.split)
		self.indices = open(split_f, 'r').read().splitlines()
		self.idx = np.array(range(self.sample_size))

		random.seed(self.seed)
		for id in range(len(self.idx)):
			self.idx[id] = random.randint(0, len(self.indices)-1)


	def set_db(self):
		params = eval(self.param_str)
		self.USE_PREFETCH = params.get('USE_PREFETCH',False)
		self.root_dir = params['root']
		self.split = params['split']
		split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.voc_dir,
				self.split)
		self.indices = open(split_f, 'r').read().splitlines()
		if USE_PREFETCH:
			self._blob_queue = Queue(10)
			self._prefetch_process = Fetcher(self._blob_queue,
											self.root_dir,
											self.indices,
											self.shape,
											self.mean,
											self.roi_num,
											self.roi_dim,
											self.seed,
											self.idx)
			self._prefetch_process.start()
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)


	def reshape(self, bottom, top):
		# implimented in forward
		pass

	def get_next_minibatch_idx(self):
		idx = np.zeros((self.shape[0]))
		for id in range(self.shape[0]):
			idx[id] = random.randint(0, len(self.indices)-1)
		return idx

	def get_next_minibatch(self):
		data_= np.zeros(self.shape)
		label_ = np.zeros((self.shape[0],1,self.shape[2],self.shape[3]))
		rois_ = np.zeros((self.roi_num,5))
		"""return everything"""
		if USE_PREFETCH:
			return self._blob_queue.get()
		else:
			idx_ = get_next_minibatch_idx()
			for id in range(self.shape[0]):
				image, label= \
					db_roi.load_imagelabel_ac(self.root_dir, self.indices[self.idx[id]], self.shape, self.mean)
				data_[id*10:(id+1)*10], label_[id*10:(id+1)*10], rois_[id*10:(id+1)*10] = \
					db_roi.getrois(id, label, self.roi_num, self.roi_dim)
		blobs['data'] = data_
		blobs['label'] = label_
		blobs['rois'] = rois_	
		return blobs


	
	def forward(self, bottom, top):
		blobs = get_next_minibatch()
		data_ = blobs['data']
		label_ = blobs['label']
		rois_ = blobs['rois']

		top[0].reshape(*data_.shape)
		top[1].reshape(*label_.shape)
		top[2].reshape(*rois_.shape)

        top[0].data[...] = data_
        top[1].data[...] = label_
        top[2].data[...] = rois_

    
    def backward(self, top, propagate_down, bottom):
        pass



class Fetcher(Process):
	"""Fetchering everything in a separate process."""
	def __init__(self, queue, root_dir, indices, shape, mean, roi_num, roi_dim, seed, idx):
		super(Fetcher, self).__init__()
		self._queue = queue
		self._root_dir = root_dir
		self._indices = indices
		self._shape = shape
		self._mean = mean
		self._roi_num = roi_num
		self._roi_dim = roi_num
		self._idx = idx;
		random.seed(seed)
		

	def get_next_minibatch_idx(self):
		idx = np.zeros((self.shape[0]))
		for id in range(self.shape[0]):
			idx[id] = random.randint(0, len(self.indices)-1)
		return idx

	def run(self):
        print 'Fetcher started'
        while True:
			_idx = get_next_minibatch_idx()
			for id in range(self.shape[0]):
				image, label= \
					db_roi.load_imagelabel_ac(self._root_dir, self._indices[_idx[id]], self._shape, self._mean)
				data_[id*10:(id+1)*10], label_[id*10:(id+1)*10], rois_[id*10:(id+1)*10] = \
					db_roi.getrois(id, label, self._roi_num, self._roi_dim)
		blobs['data'] = data_
		blobs['label'] = label_
		blobs['rois'] = rois_	
		self._queue.put(blobs)