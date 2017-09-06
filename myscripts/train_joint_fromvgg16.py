import caffe
import numpy as np
import scipy.io
import math
import os
import sys
from PIL import Image
sys.path.append("..")
sys.path.append("../pylayers")
from pyutils import refine_util as rv
from pyutils import surgery as sg

def init_train():
	caffe.set_mode_gpu()
	caffe.set_device(0)
	solverproto = '../models/voc_joint/solver_v{}.prototxt'.format(sys.argv[1])
	Sov = rv.parse_solverproto(solverproto)
	if not os.path.exists('../states'):
		os.makedirs('../states')
	save_path = '../states/joint_v{}/'.format(sys.argv[1])	
	if not os.path.exists(save_path):
		os.makedirs(save_path)	
	solver = caffe.SGDSolver(solverproto)
	solver.set_iter(0)
	max_iter = 30001;
	save_iter = 10;
	display_iter = 10
	# train_.train_loss = 0
	tmpname = save_path + 'loss' + '.mat'
	cur_res_g = save_path+'infer_res_g.mat'
	cur_res_s = save_path+'infer_res_s.mat'
	cur_iter = save_path+'iter.mat'
	train_ = {'save_path':save_path, 'max_iter':max_iter, 'save_iter':save_iter, 'display_iter':display_iter, 'tmpname':tmpname, 'cur_res_g':cur_res_g, 'cur_res_s':cur_res_s, 'cur_iter':cur_iter}
	return solver, Sov, train_

if __name__ == "__main__":
	solver, Sov, train_ = init_train()
	if not os.path.exists(train_['cur_iter']):
		# apply global model (VGG-16)
		weights_global = '../models/pretrain/train_iter_20000.caffemodel'
		proto_global = '../models/voc_joint/deeplabv2-vgg16-deploy.prototxt'
		net_old = caffe.Net(proto_global, weights_global, caffe.TRAIN)
		sg.transplant(solver.net, net_old, '_global')

		# apply SPN model (VGG-convs)
		weights_guide = '../models/pretrain/vgg16_20M.caffemodel'
		solver.net.copy_from(weights_guide)

		# loss zeros
		train_loss_g = np.zeros(int(math.ceil(train_['max_iter']/ train_['display_iter'])))
		train_loss_s = np.zeros(int(math.ceil(train_['max_iter']/ train_['display_iter'])))
	else:
		curiter = scipy.io.loadmat(train_['cur_iter'])
		curiter = curiter['cur_iter']
		curiter = int(curiter)
		solver.set_iter(curiter)
		train_loss = scipy.io.loadmat(train_['tmpname'])
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
	# _train_loss_g = 0
	_train_loss_s = 0

	for iter in range(begin, train_['max_iter']):

		solver.step(1)
		# _train_loss_g += solver.net.blobs['loss_global'].data
		_train_loss_s += solver.net.blobs['loss'].data

		if iter % train_['display_iter'] == 0:
			# train_loss_g[int(iter / train_['display_iter'])] = _train_loss_g / train_['display_iter']
			train_loss_s[int(iter / train_['display_iter'])] = _train_loss_s / train_['display_iter']
			# _train_loss_g = 0
			_train_loss_s = 0

		if iter % train_['save_iter'] == 0:

			batch, label, global_prob = rv.get_global(solver.net)
			scipy.io.savemat(train_['cur_res_g'], dict(batch = batch, label = label, global_prob = global_prob))

			batch_rois, label_rois, msk_rois, spn_active = rv.get_spn(solver.net)
			scipy.io.savemat(train_['cur_res_s'], dict(batch_rois = batch_rois, label_rois = label_rois, msk_rois = msk_rois, spn_active = spn_active))

			# DEBUG
			batch_rois, label_rois, msk_rois, spn_active = rv.get_spn(solver.net)
			scipy.io.savemat('{}/debug_active.mat'.format(train_['save_path']), dict(batch_rois = batch_rois, label_rois = label_rois, msk_rois = msk_rois, spn_active = spn_active))

			batch_diff, prob_diff, msk_diff, spn_diff = rv.get_spn_diff(solver.net)
			scipy.io.savemat('{}/debug_diff.mat'.format(train_['save_path']), dict(batch_diff = batch_diff, prob_diff = prob_diff, msk_diff = msk_diff, spn_diff = spn_diff))

			scipy.io.savemat(train_['cur_iter'], dict(cur_iter = iter))	
			# scipy.io.savemat(train_['tmpname'], dict(train_loss_g = train_loss_g, train_loss_s = train_loss_s))
			scipy.io.savemat(train_['tmpname'], dict(train_loss_s = train_loss_s))

		if (iter-1) % Sov['snapshot'] == 0:
			rv.clear_history(Sov['snapshot'],Sov['snapshot_prefix'],iter-1)