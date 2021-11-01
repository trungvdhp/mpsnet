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
        
    def _norm_layer(self, inputs, norm='batch', name=None):
        if(norm=='batch'):
            Z = BatchNormalization(axis=self.channel_axis, name=name)(inputs)
        else:
            Z = inputs
        return Z
    
    def make_divisible(self, v, divisor, min_value=None):
        # This function is taken from the original tf repo.
        # It ensures that all layers have a channel number that is divisible by 8
        # It can be seen here:
        # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    def _hard_swish(self, x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0
    
    def _squeeze(self, inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        input_channels = int(inputs.shape[channel_axis])

        x = GlobalAveragePooling2D()(inputs)
        x = Reshape((1, 1, input_channels))(x)
        x = conv_block(x, input_channels//4, 1, 1, padding='same', norm=None, activation='relu')
        x = conv_block(x, input_channels, 1, 1, padding='same', norm=None, activation='hard_sigmoid')
        x = Multiply()([inputs, x])

        return x
    
    def _activation_layer(self, x, activation='relu'):
        if activation == 'hard_sigmoid':
            x = Activation(activation)(x)
        if activation == 'hard_swish':
            x = Activation(_hard_swish)(x)
        if activation == 'relu':
            x = ReLU()(x)
        if activation == 'relu6':
            x = ReLU(max_value=6.0)(x)
        if activation == 'prelu':
            x = PReLU(shared_axes = self.shared_axes)(x)

        return x

    def conv_block(self, inputs, filters, kernel_size, strides, padding='same', norm='batch', activation='relu', dropout=0.0):

        x = Conv2D(filters, kernel_size, strides = strides, 
                   padding = padding, use_bias = self.use_bias,
                   kernel_initializer = self.kernel_initializer,
                   kernel_regularizer = self.kernel_regularizer)(inputs)
       
        x = self._norm_layer(x, norm=norm)
        x = self._activation_layer(x, activation=activation)
        
        if(dropout>0):
            x = Dropout(rate=dropout)(x)
            
        return x

    def separable_conv_block(self, inputs, filters, kernel_size, strides, padding='same', norm='batch', activation='relu', dropout=0.0):

        x = SeparableConv2D(filters, kernel_size, strides = strides, 
                            padding = padding, use_bias = self.use_bias,
                            kernel_initializer = self.kernel_initializer,
                            kernel_regularizer = self.kernel_regularizer
                           )(inputs)
        x = self._norm_layer(x, norm=norm)
        x = self._activation_layer(x, activation=activation)
            
        if(dropout>0):
            x = Dropout(rate=dropout)(x)
        return x

    def depthwise_conv_block(self, inputs, kernel_size, strides, padding='same', norm='batch', activation='relu', dropout=0.0):
        
        x = DepthwiseConv2D(kernel_size, strides = strides, padding = padding, 
                            depth_multiplier = 1, use_bias = self.use_bias,
                            kernel_initializer = self.kernel_initializer,
                            kernel_regularizer = self.kernel_regularizer
                           )(inputs)

        x = self._norm_layer(x, norm=norm)
        x = self._activation_layer(x, activation=activation)
        
        if(dropout>0):
            x = Dropout(rate=dropout)(x)
            
        return x
    
    def spp_block(self, x, dropout=0.2, pool_list=[1, 2, 4], pool_type='max', norm='batch'):
        if(dropout>0):
            x = Dropout(rate = dropout)(x)
        x = SpatialPyramidPooling(pool_list=pool_list, pool_type=pool_type)(x)
        x = self._norm_layer(x, norm=norm)
        
        return x

    def bottleneck_v1(self, inputs, c, ks, s, alpha=1.0, norm='batch', activation='relu'):
        
        c = int(c*alpha)
        
        x = self.depthwise_conv_block(inputs, ks, s, padding='same', norm=norm, activation=activation)
        x = self.conv_block(x, c, kernel_size=1, strides=1, norm=norm, activation=activation)

        return x

    def bottleneck_v2(self, inputs, c, ks, t, s, r = False, norm='batch', activation='relu6'):
        
        tchannel = K.int_shape(inputs)[self.channel_axis] * t
        
        x = self.conv_block(inputs, tchannel, kernel_size=1, strides=1, padding='same', norm=norm, activation=activation)
        x = self.depthwise_conv_block(x, ks, s, padding='same', norm=norm, activation=activation)
        x = self.conv_block(x, c, kernel_size=1, strides = 1, padding = 'same', norm=norm, activation=None)
        
        if r:
            x = add([x, inputs])
        
        return x

    def inverted_residual_block(self, inputs, c, ks, t, s, n, norm='batch', activation='relu', alpha=1.0):
        
        c = self.make_divisible(c*alpha, 8)
        x = self.bottleneck_v2(inputs, c, ks, t, s, r=False, norm=norm, activation=activation)
        
        for i in range(1, n):
            x = self.bottleneck_v2(x, c, ks, t, 1, r=True, norm=norm, activation=activation)
        
        return x
    
    def bottleneck_v3(self, inputs, c, ks, t, s, alpha=1.0, squeeze=False, norm='batch', activation='relu'):

        input_shape = K.int_shape(inputs)
        tchannel = int(t)
        cchannel = int(alpha*filters)
        r = s==1 and input_shape[channel_axis]==c

        x = self.conv_block(inputs, tchannel, 1, 1, norm=norm, activation=activation)
        x = self.depthwise_conv_block(x, ks, s, padding='same', norm=norm, activation=activation)

        if squeeze:
            x = self._squeeze(x)

        x = self.conv_block(x, cchannel, kernel_size=1, strides=1, norm=norm, activation=None)

        if r:
            x = Add()([x, inputs])

        return x

    
