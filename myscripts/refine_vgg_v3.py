import caffe
import numpy as np
import scipy.io
import math
import os
from PIL import Image
import sys
sys.path.append("..")
sys.path.append("../pylayers")
from pyutils import refine_util as rv

if __name__ == "__main__":
	caffe.set_mode_gpu()
	caffe.set_device(0)
	solverproto = '../models/voc_joint/solver_v3.prototxt'
	solver = caffe.SGDSolver(solverproto)
	# default 0, need to be set when restart
	solver.set_iter(0) 
	# =============================
	Sov = rv.parse_solverproto(solverproto)
	max_iter = 60000;
	save_iter = 100;
	save_path = '../states/joint_v3/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	display_iter = 10
	_train_loss = 0
	tmpname = save_path + 'loss' + '.mat'
	weights = '../models/pretrain/vgg16_20M.caffemodel'
	solver.net.copy_from(weights)
	
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	if solver.iter == 0:
		solver.step(1)
		solver.set_iter(0)
		train_loss = np.zeros(math.ceil(max_iter/ display_iter))
	else:
		train_loss = scipy.io.loadmat(tmpname)
		train_loss = np.array(train_loss['train_loss'], dtype=np.float32).squeeze()
		solverstate = Sov['snapshot_prefix'] + \
		'_iter_{}.solverstate'.format(solver.iter)
		caffemodel = Sov['snapshot_prefix'] + \
		'_iter_{}.caffemodel'.format(solver.iter)
		if 'solverstate' in locals():
			solver.restore(solverstate)
		elif 'caffemodel' in locals():
			solver.net.copy_from(caffemodel)
		else:
			raise Exception("Model does not exist.")
	begin = solver.iter

	for iter in range(begin, max_iter):
		solver.step(1)
		_train_loss += solver.net.blobs['loss'].data
		if iter % display_iter == 0:
			train_loss[iter / display_iter] = _train_loss / display_iter
			_train_loss = 0
		if iter % save_iter == 0:
			batch, label, active = rv.getbatch(solver.net)
			cur_res_mat = save_path +'infer_res.mat'
			scipy.io.savemat(cur_res_mat, dict(batch = batch, label = label, active = active))	
			scipy.io.savemat(tmpname, dict(train_loss = train_loss))
		if (iter-1) % Sov['snapshot'] == 0:
			rv.clear_history(Sov['snapshot'],Sov['snapshot_prefix'],iter)
