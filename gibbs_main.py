import sys

import chainer
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import cPickle as pickle



patch_size = 15
parentpath= '/home/koyama-m/Research/membrane_CNN/'
models_path='/home/koyama-m/Research/membrane_CNN/models/'

sys.path.append(parentpath)
sys.path.append(models_path)

reconstruction_path = '/home/koyama-m/Research/membrane_CNN/data/reconstructed_256images_crop15'
probmap_prefix = 'multi_crop_prediction_image_256_'
#multi_crop_prediction_image_256_001.tif
binmap_prefix = 'multi_crop_prediction_binary_image_256_' 
#multi_crop_prediction_binary_image_256_001.tif
#hole0_cool_rate0.95conditional_distr_trained_model256_crop15epoch100.pkl
modelname = 'hole0_cool_rate0.95conditional_distr_trained_model256_crop15epoch1.pkl'
modelname = 'trained_model256_crop15epoch100.pkl'

print models_path +modelname

model = pickle.load(open(models_path +modelname ))
