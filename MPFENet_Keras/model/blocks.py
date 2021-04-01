import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from tensorflow.keras import regularizers
from model.spp import SpatialPyramidPooling

class NetBlock:
    def __init__(self, config):
        self.use_bias = False if config.use_bias==0 else True
        self.weight_decay = config.weight_decay
        self.kernel_initializer = config.kernel_initializer
        self.kernel_regularizer = regularizers.l2(self.weight_decay)
        self.channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        self.shared_axes = [2, 3] if K.image_data_format() == 'channels_first' else [1, 2]
        
    def norm_layer(self, inputs, norm='batch', name=None):
        if(norm=='batch'):
            Z = BatchNormalization(axis=self.channel_axis, name=name)(inputs)
        else:
            Z = inputs
        return Z

    def conv_block(self, inputs, filters, kernel_size, strides, padding='same', norm='batch', activation='prelu', relu_max_value=None, dropout=0.0, name=None):

        Z = Conv2D(filters, kernel_size, strides = strides, 
                   padding = padding, use_bias = self.use_bias,
                   kernel_initializer = self.kernel_initializer,
                   kernel_regularizer = self.kernel_regularizer,
                   name=name
                  )(inputs)
       
        Z = self.norm_layer(Z, norm, name=None)
        
        if(activation=='prelu'):
            Z = PReLU(shared_axes = self.shared_axes)(Z)
        elif(activation=='relu'):
            Z = ReLU(max_value=relu_max_value)(Z)
        elif(activation=='softmax'):
            Z = Activation(activation, name=activation)(Z)
        
        if(dropout>0):
            Z = Dropout(rate=dropout)(Z)
        return Z

    def separable_conv_block(self, inputs,filters,kernel_size,strides,padding='same',norm='batch', activation='prelu', relu_max_value=None, dropout=0.0):

        Z = SeparableConv2D(filters, kernel_size, strides = strides, 
                            padding = padding, use_bias = self.use_bias,
                            kernel_initializer = self.kernel_initializer,
                            kernel_regularizer = self.kernel_regularizer,
                           )(inputs)
        Z = self.norm_layer(Z, norm, name=None)
        
        if(activation=='prelu'):
            Z = PReLU(shared_axes = self.shared_axes)(Z)
        elif(activation=='relu'):
            Z = ReLU(max_value=relu_max_value)(Z)
        elif(activation=='softmax'):
            Z = Activation(activation, name=activation)(Z)
            
        if(dropout>0):
            Z = Dropout(rate=dropout)(Z)
        return Z

    def spp_block(self, inputs, dropout=0.2, pool_list=[1, 2, 4], norm='batch'):
        M = Dropout(rate = dropout)(inputs)
        M = SpatialPyramidPooling(pool_list=pool_list, pool_type='max')(M)
        M = self.norm_layer(M, norm=norm)
        
        return M

    def bottleneck(self, inputs, c, ks, t, s, r = False, norm='batch', activation='relu', relu_max_value=None):
        
        tchannel = K.int_shape(inputs)[self.channel_axis] * t
        
        Z1 = self.conv_block(inputs, tchannel, kernel_size=1, strides=1, padding='same', activation=activation,norm=norm)
        
        Z1 = DepthwiseConv2D(ks, strides=s, padding='same', depth_multiplier=1, 
                             use_bias=self.use_bias,
                             kernel_initializer = self.kernel_initializer,
                             kernel_regularizer = self. kernel_regularizer
                            )(Z1)
        Z1 = self.norm_layer(Z1, norm, name=None)
        
        if(activation=='prelu'):
            A1 = PReLU(shared_axes = self.shared_axes)(Z1)
        elif(activation=='relu'):
            A1 = ReLU(max_value=relu_max_value)(Z1)
        
        Z2 = self.conv_block(A1, c, kernel_size=1, strides = 1, padding = 'same', activation=None,norm=norm)
        
        if r:
            Z2 = add([Z2, inputs])
        
        return Z2

    def inverted_residual_block(self, inputs, c, ks, t, s, n, norm='batch', activation='relu', relu_max_value=None):
        
        Z = self.bottleneck(inputs, c, ks, t, s, r=False, norm=norm, activation=activation)
        
        for i in range(1, n):
            Z = self.bottleneck(Z, c, ks, t, 1, r=True, norm=norm, activation=activation)
        
        return Z

    def depthwise_conv_block(self, inputs, kernel_size, strides, norm='batch', dropout=0.0, name=None):
        
        Z = DepthwiseConv2D(kernel_size, strides = strides, 
                            padding = 'valid', depth_multiplier = 1,
                            use_bias = self.use_bias,
                            kernel_initializer = self.kernel_initializer,
                            kernel_regularizer = self.kernel_regularizer,
                            name = name
                           )(inputs)

        Z = self.norm_layer(Z, norm=norm)
        
        if(dropout>0):
            Z = Dropout(rate=dropout)(Z)
            
        return Z
