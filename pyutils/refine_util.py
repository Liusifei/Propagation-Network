import caffe
import numpy as np
import scipy.io
import math
import os
from PIL import Image
# import vocaugcropsoft_layer as VL

def imresize4array(imarray, width, height):
	if imarray.ndim == 2:
		imarray = np.array(imarray,dtype = np.uint8)
	elif imarray.ndim == 3:
		imarray = np.array(imarray,dtype = np.float32)
	im = Image.fromarray(imarray)
	im = im.resize((width,height), resample=Image.BILINEAR)
	im = np.array(im, dtype=np.float32)
	return im


def getbatch(net):
	batch = net.blobs['data'].data[...]
	active = net.blobs['deconv0'].data[...]
	label = net.blobs['label'].data[...]
	return (batch, label, active)


# important checkpoints
def get_global(net):
	batch = net.blobs['image'].data[...]
	global_prob = net.blobs['prob'].data[...]
	label = net.blobs['label'].data[...]
	return (batch, label, global_prob)


def get_spn(net):
	batch_rois = net.blobs['rgb'].data[...]
	label_rois = net.blobs['labels'].data[...]
	msk_rois = net.blobs['msk'].data[...]
	spn_active = net.blobs['deconv0'].data[...]
	return (batch_rois, label_rois, msk_rois, spn_active)

def get_spn_diff(net):
	batch_diff = net.blobs['conv1_1'].diff[...]
	prob_diff = net.blobs['prob'].diff[...]
	msk_diff = net.blobs['msk'].diff[...]
	spn_diff = net.blobs['fc8_interp'].diff[...]
	return (batch_diff, prob_diff, msk_diff, spn_diff)

# more general
def get_alldata(net):
	data_all = {}
	for a in net.blobs:
		data_all[a] = net.blobs[a].data[...]
	return data_all

def get_alldiff(net):
	diff_all = {}
	for a in net.blobs:
		diff_all[a] = net.blobs[a].diff[...]
	return diff_all

def get_all_weights(net):
	weights_all = {}
	for p in net.params:
		weights_ = {}
		for i in range(len(net.params[p])):
			weights_[i] = net.params[p][i].data[...]
		weights_all[p] = weights_
	return weights_all

def clear_history(snapshot,snapshot_prefix,iter):
	if iter > 2*snapshot and (iter-snapshot)%10000!=0:
		defile = snapshot_prefix + '_iter_{}.caffemodel'.format(iter-2*snapshot)
		if os.path.isfile(defile):
			os.remove(defile)
		defile = snapshot_prefix + '_iter_{}.solverstate'.format(iter-2*snapshot)
		if os.path.isfile(defile):
			os.remove(defile)

def parse_solverproto(solverproto):
	solverfile = open(solverproto,'r').read().splitlines()
	net = solverfile[0][6:-1]
	snapshot = next(s for s in solverfile if 'snapshot:' in s)
	snapshot = np.array(snapshot[10:],dtype = np.uint16)
	snapshot_prefix = next(s for s in solverfile if 'snapshot_prefix:' in s)
	snapshot_prefix = snapshot_prefix[18:-1]
	return {'net': net, 'snapshot': snapshot, 'snapshot_prefix': snapshot_prefix}
