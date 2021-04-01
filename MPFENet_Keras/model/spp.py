import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

class SpatialPyramidPooling(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if image_data_format = 'channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if image_data_format = 'channels_last'
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, pool_type, **kwargs):
        self.pool_list = pool_list
        self.pool_type = pool_type
        super(SpatialPyramidPooling, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        if(K.image_data_format()=='channels_first'):
            num_channels = input_shape[1]
        else:
            num_channels = input_shape[3]
        num_output_per_channel = sum([i * i for i in self.pool_list])   
        output_channels = num_channels * num_output_per_channel
        
        return (input_shape[0], output_channels)

    def get_config(self):
        config = {'pool_list': self.pool_list,
                  'pool_type': self.pool_type
                 }
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):

        image_data_format = K.image_data_format()
        input_shape = K.int_shape(inputs)
        outputs = []
        epsilon = K.epsilon()

        if image_data_format == 'channels_first':
            
            for pool_size in self.pool_list:
                
                win_size = input_shape[2]//pool_size
                
                for jy in range(pool_size):
                    
                    for ix in range(pool_size):
                        
                        x = int(ix*win_size)
                        y = int(jy*win_size)
                        
                        if(self.pool_type=='max'):
                            pooled_vals = K.max(inputs[:,:,x:x+win_size,y:y+win_size], axis=(2, 3))
                        elif(self.pool_type=='min'):
                            pooled_vals = K.min(inputs[:,:,x:x+win_size,y:y+win_size], axis=(2, 3))
                        elif(self.pool_type=='avg'):
                            pooled_vals = K.mean(inputs[:,:,x:x+win_size,y:y+win_size], axis=(2, 3))
                        else:
                            if(win_size==1):
                                pooled_vals = K.max(inputs[:,:,x:x+win_size,y:y+win_size], axis=(2, 3))
                            else:
                                max_vals = K.max(inputs[:,:,x:x+win_size,y:y+win_size], axis=(2, 3))
                                mean_vals = K.mean(inputs[:,:,x:x+win_size,y:y+win_size], axis=(2, 3))
                                std_vals = K.std(inputs[:,:,x:x+win_size,y:y+win_size], axis=(2, 3)) + epsilon
                                pooled_vals = (max_vals-mean_vals)/std_vals
                                print(win_size, K.int_shape(pooled_vals))
                        
                        outputs.append(pooled_vals)
        elif image_data_format == 'channels_last':
            
            for pool_size in self.pool_list:

                win_size = input_shape[2]//pool_size
                
                for jy in range(pool_size):
                    
                    for ix in range(pool_size):
                        
                        x = int(ix*win_size)
                        y = int(jy*win_size)
                        pooled_val = K.max(inputs[:,x:x+win_size,y:y+win_size,:], axis=(1, 2))
                        outputs.append(pooled_val)

        if image_data_format == 'channels_first':
            outputs = K.concatenate(outputs)
        elif image_data_format == 'channels_last':
            #outputs = K.concatenate(outputs,axis = 1)
            outputs = K.concatenate(outputs)
            #outputs = K.reshape(outputs,(len(self.pool_list),self.num_outputs_per_channel,input_shape[0],input_shape[1]))
            #outputs = K.permute_dimensions(outputs,(3,1,0,2))
            #outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))

        return outputs