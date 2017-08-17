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