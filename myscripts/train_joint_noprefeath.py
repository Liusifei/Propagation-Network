import caffe
import numpy as np
import scipy.io
import math
import os
import sys
from PIL import Image
sys.path.append("..")
from pyutil import refine_util as rv
from pyutil import surgery as sg

def init_train():
	caffe.set_mode_gpu()
	caffe.set_device(0)
	solverproto = '../models/voc_joint/solver_v{}.prototxt'.format(sys.argv[1])
	Sov = rv.parse_solverproto(solverproto)
	solver = caffe.SGDSolver(solverproto)
	solver.set_iter(0)
	train_.max_iter = 20001;
	train_.save_iter = 100;
	train_.display_iter = 10
	# train_.train_loss = 0
	train_.tmpname = save_path + 'loss' + '.mat'
	train_.cur_res_mat = save_path+'infer_res.mat'
	train_.cur_iter = save_path+'iter.mat'
	train_.save_path = '../states/joint_v{}/'.format(sys.argv[1])
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	return solver, Sov, train_

if __name__ == "__main__":
	solver_, Sov_, train_ = init_train()
	if not os.path.exists(train_.cur_iter):
		weights_global = '../models/pretrain/train_iter_20000.caffemodel'
		weights_guide = '../models/pretrain/vgg_scratch_c7_iter_60000.caffemodel'
		solver.net.copy_from(weights_global)
		solver.net.copy_from(weights_guide)
		train_loss = np.zeros(int(math.ceil(max_iter/ display_iter)))
	else:
		curiter = scipy.io.loadmat(train_.cur_iter)
		curiter = curiter['cur_iter']
		curiter = int(curiter)
		solver.set_iter(curiter)
		train_loss = scipy.io.loadmat(train_.tmpname)
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
	_train_loss = 0

	for iter in range(begin, max_iter):
		solver.step(1)
		_train_loss += solver.net.blobs['loss'].data
		if iter % display_iter == 0:
			train_loss[int(iter / display_iter)] = _train_loss / display_iter
			_train_loss = 0
		if iter % save_iter == 0:
			batch, label, active = rv.getbatch(solver.net)
			scipy.io.savemat(cur_res_mat, dict(batch = batch, label = label, active = active))
			scipy.io.savemat(train_.cur_iter, dict(cur_iter = iter))	
			scipy.io.savemat(train_.tmpname, dict(train_loss = train_loss))
		if (iter-1) % Sov['snapshot'] == 0:
			rv.clear_history(Sov['snapshot'],Sov['snapshot_prefix'],iter-1)