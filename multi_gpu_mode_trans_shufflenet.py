#-*- coding:utf-8 -*-
import os
import sys
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras.layers import Input
from keras.models import Model
# import keras.backend as K
from keras.utils import multi_gpu_model
#import dl_resnet_crnn_cudnnlstm as densenet
import shufflenet_res_crnn as densenet
#import densenet
os.environ["CUDA_VISIBLE_DEVICES"] = "7,6"
GPU_NUM = 2
#reload(densenet)

encode_dct =  {}
#char_set = open('japchn.txt', 'r', encoding='utf-8').readlines()
char_set = open('japeng.txt', 'r', encoding='utf-8').readlines()
for i in range (0, len(char_set)):
	c = char_set[i].strip('\n')
	encode_dct[c] = i

char_set.append('卍')
#char_set = ''.join([ch.strip('\n') for ch in char_set] + ['卍'])

#characters = ''.join([chr(i) for i in range(32, 127)] + ['卍'])
nclass = len(char_set)


mult_model, basemodel = densenet.get_model(False, 32, nclass)
#input = Input(shape=(32, None, 1), name='the_input')
#y_pred= densenet.dense_cnn(input, nclass)
#basemodel = Model(inputs=input, outputs=y_pred)

#model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)

mp = 'weights_densenet-02-1.29.h5'
modelPath = os.path.join(os.getcwd(), './models/' + mp)
modelPath = sys.argv[1]
'''
load models
'''
#basemodel.load_weights("./new_model.h5")



if os.path.exists(modelPath):
    multi_model = multi_gpu_model(basemodel, gpus=GPU_NUM)
    multi_model.load_weights(modelPath)
    basemodel.save(sys.argv[2])
    #basemodel.save("./new_model.h5")
    #basemodel = multi_model
    #model.load_weights(modelPath)

