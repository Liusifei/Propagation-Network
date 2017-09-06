import caffe
import numpy as np
import os
from PIL import Image
import sys
sys.path.append("..")
from pyutils import score_net as score

"""
for VOC
sys.argv[1]: deploy_XXX.prototxt
sys.argv[2]: e.g., joint_v1
sys.arge[3]: iter
sys.arge[4]: layer name
"""
test_proto = '../models/voc_joint/deploy-{}.prototxt'.format(sys.argv[1])
test_model = '../states/{}/state_iter_{}.caffemodel'.format(sys.argv[2],sys.argv[3])
if len(sys.argv) > 4:
	layer = sys.argv[4]
else:
	layer = 'prob'
dataroot = '/home/sifeil/Data/VOC/pascal/VOCdevkit/VOC2012'
# prior_root = '/media/sifeil/NV_share/Results/davis_global_JC3/ResNetF_perobj_27000/'
result_folder = '/home/sifeil/voc_results/{}_{}'.format(sys.argv[2], sys.argv[4])
caffe.set_device(0)
caffe.set_mode_gpu()
cnnnet = caffe.Net(test_proto, test_model, caffe.TEST)
val = np.loadtxt('{}/ImageSets/Segmentation/val.txt'.format(dataroot), dtype=str)
score.seg_tests(cnnnet, sys.argv[3], dataroot, result_folder, val, layer, gt='label')