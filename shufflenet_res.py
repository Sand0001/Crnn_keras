# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Tue Oct 23 14:03:51 2018

@author: xingshuli
"""

# from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import DepthwiseConv2D
# from keras.applications.mobilenet import DepthwiseConv2D
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras import backend as K
import numpy as np


def channel_split(x, name=''):
    in_channels = x.shape.as_list()[-1]
    ip = in_channels // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip])(x)
    c = Lambda(lambda z: z[:, :, :, ip:])(x)

    return c_hat, c


def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2

    #x = K.reshape(x, [-1, -1, width, 2, channels_per_split])
    x = Reshape([-1, width, 2, channels_per_split])(x)
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    #x = K.reshape(x, [-1, -1, width, channels])
    x = Reshape( [ -1, width, channels])(x)
    return x


def _shuffle_unit(inputs, out_channels, strides=2, stage=1, block=1):
    bn_axis = -1
    prefix = 'stage%d/block%d' % (stage, block)

    branch_channels = out_channels // 2
    # branch_channels = 512

    if stage == 4:

        strides_4 = 1
    elif stage == 3:
        strides_4 = 2
    elif stage == 2:
        strides_4 = 1
    else:
        strides_4 = 2

    if strides == 2:
        x_1 = DepthwiseConv2D(kernel_size=3, strides=strides_4, padding='same',
                              use_bias=False, name='%s/3x3dwconv_1' % prefix)(inputs)
        x_1 = BatchNormalization(axis=bn_axis, name='%s/bn_3x3dwconv_1' % prefix)(x_1)
        x_1 = Conv2D(filters=branch_channels, kernel_size=1, strides=1, padding='same',
                     use_bias=False, name='%s/1x1conv_1' % prefix)(x_1)
        x_1 = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_1' % prefix)(x_1)
        x_1 = Activation('relu')(x_1)

        x_2 = Conv2D(filters=branch_channels, kernel_size=1, strides=1, padding='same',
                     use_bias=False, name='%s/1x1conv_2' % prefix)(inputs)
        x_2 = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_2' % prefix)(x_2)
        x_2 = Activation('relu')(x_2)
        x_2 = DepthwiseConv2D(kernel_size=3, strides=strides_4, padding='same',
                              use_bias=False, name='%s/3x3dwconv_2' % prefix)(x_2)
        x_2 = BatchNormalization(axis=bn_axis, name='%s/bn_3x3dwconv_2' % prefix)(x_2)
        x_2 = Conv2D(filters=branch_channels, kernel_size=1, strides=1, padding='same',
                     use_bias=False, name='%s/1x1conv_3' % prefix)(x_2)
        x_2 = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_3' % prefix)(x_2)
        x_2 = Activation('relu')(x_2)

        x = Concatenate(axis=bn_axis, name='%s/concat' % prefix)([x_1, x_2])

    if strides == 1:
        c_hat, c = channel_split(inputs, name='%s/split' % prefix)

        c = Conv2D(filters=branch_channels, kernel_size=1, strides=1, padding='same',
                   use_bias=False, name='%s/1x1conv_4' % prefix)(c)
        c = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_4' % prefix)(c)
        c = Activation('relu')(c)
        c = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
                            use_bias=False, name='%s/3x3dwconv_3' % prefix)(c)
        c = BatchNormalization(axis=bn_axis, name='%s/bn_3x3dwconv_3' % prefix)(c)
        c = Conv2D(filters=branch_channels, kernel_size=1, strides=1, padding='same',
                   use_bias=False, name='%s/1x1conv_5' % prefix)(c)
        c = BatchNormalization(axis=bn_axis, name='%s/bn_1x1conv_5' % prefix)(c)
        c = Activation('relu')(c)

        x = Concatenate(axis=bn_axis, name='%s/concat' % prefix)([c_hat, c])

    x = Lambda(channel_shuffle, name='%s/channel_shuffle' % prefix)(x)

    return x


def v2_block(x, channel_map, repeat=1, stage=1):
    x = _shuffle_unit(x, out_channels=channel_map[stage - 1], strides=2,
                      stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = _shuffle_unit(x, out_channels=channel_map[stage - 1], strides=1,
                          stage=stage, block=(i + 1))

    return x

def res_block(x):
    x = res18_conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(1, 2))  # height // 2
    x = res18_identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    #x = resnet.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    # x = resnet.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    # x = resnet.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    # x = resnet.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    return x

def res18_identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    # x = Activation('Mish')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(input_tensor)

    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)


    # x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def res18_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    #size // 2 减少步长
    # x = Conv2D(filters1, (1, 1), strides=strides,
    #            name=conv_name_base + '2a')(input_tensor)
    # x = Activation('Mish')(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)


    x = Conv2D(filters2, kernel_size,strides=strides, padding='same',
               name=conv_name_base + '2b')(input_tensor)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)


    # x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters2, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ShuffleNet_V2(include_top=True, input_tensor=None, scale_factor=1.0, pooling='avg',
                  input_shape=(224, 224, 3), num_shuffle_units=[3, 7, 3], weights=None,
                  classes=1000):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only TensorFlow backend is currently supported')

    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size = 224,
    #                                   min_size = 28,
    #                                   data_format = K.image_data_format(),
    #                                   require_flatten = include_top,
    #                                   weights = weights)
    input_shape = (280, 32, 1)

    out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')

    if not (float(scale_factor) * 4).is_integer():
        raise ValueError('Invalid value for scale_factor. Should be x over 4')

    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[scale_factor]
    out_channels_in_stage[0] = 24
    out_channels_in_stage = out_channels_in_stage.astype(int)
    print('out_channels_in_stage', out_channels_in_stage)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(shape=input_shape, tensor=input_tensor)
        else:
            img_input = input_tensor

    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=3, strides=1,
               padding='same', use_bias=False, activation='relu', name='conv1')(img_input)

    x = MaxPooling2D(pool_size=3, strides=2, padding='same', name='MaxPool1')(x)

    # construct stage2 to 4
    # stage2

    # x = v2_block(x,channel_map = out_channels_in_stage, repeat = 3, stage = 2)

    # x = _shuffle_unit(x, out_channels=channel_map[stage - 1], strides=2,
    #                   stage=stage, block=1)

    # x = _shuffle_unit(x, out_channels=116, strides=1,
    #                   stage=2, block=1)
    #
    # for i in range(1, 4 + 1):
    #     x = _shuffle_unit(x, out_channels=2, strides=1,
    #                       stage=2, block=(i + 1))
    # #stage3
    #
    #
    # x = v2_block(x, channel_map=out_channels_in_stage, repeat=7, stage=3)
    # #stage4
    #
    # x = _shuffle_unit(x, out_channels=464, strides=1,
    #                   stage=4, block=1)
    # for i in range(1, 4 + 1):
    #     x = _shuffle_unit(x, out_channels=464, strides=1,
    #                       stage=4, block=(i + 1))

    for stage in range(len(num_shuffle_units)-1):
        repeat = num_shuffle_units[stage]
        #print('repeat', repeat, stage)
        x = v2_block(x, channel_map=out_channels_in_stage, repeat=repeat, stage=stage + 2)
    #  stage4

    x = res_block(x)

    # construct final layers
    if scale_factor == 2:
        x = Conv2D(filters=2048, kernel_size=1, strides=1, padding='same',
                   use_bias=False, activation='relu', name='conv5')(x)
    else:
        x = Conv2D(filters=1024, kernel_size=1, strides=1, padding='same',
                   use_bias=False, activation='relu', name='conv5')(x)

    # if pooling == 'avg':
    #     x = GlobalAveragePooling2D(name='global_average_pool')(x)
    # elif pooling == 'max':
    #     x = GlobalMaxPooling2D(name='global_max_pool')(x)

    # if include_top:
    #     x = Dense(classes, name='fc')(x)
    #     x = Activation('softmax')(x)
    #
    # if input_tensor is not None:
    #     inputs = get_source_inputs(input_tensor)
    # else:
    #     inputs = img_input
    # 
    # # construct model function
    # model = Model(inputs=inputs, outputs=x, name='ShuffleNet_V2')
    # model.summary()
    return x


if __name__ == '__main__':
    ShuffleNet_V2()





