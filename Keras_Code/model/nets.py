from model.adacos import AdaCos
from model.blocks import NetBlock
from tensorflow.keras.layers import Input, Reshape, Conv2D, Activation, Flatten, Dropout, add
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class Net:
    
    def __init__(self, config):
        self.model_name = config.model_name
        self.start_fine_tune_layer_id = config.start_fine_tune_layer_id
        self.end_fine_tune_layer_id = config.end_fine_tune_layer_id
        self.embedding_dim = config.embedding_dim
        self.embedding_layer_name = config.embedding_layer_name
        self.dropout = config.dropout
        self.net_blocks = NetBlock(config)

    def build_mpsnet_backbone(self, input_shape):
        
        c = [32, 32, 64, 64, 128]
        t = [1, 2, 2, 3, 2]
        s = [2, 2, 2, 2, 1]
        n = [1, 2, 2, 3, 2]
        
        activation='relu'
        I = Input(shape = input_shape)
        
        M0 = self.net_blocks.conv_block(I, c[0], 3, s[0], activation=activation)
        M1 = self.net_blocks.inverted_residual_block(M0, c=c[1], ks=3, t=t[1], s=s[1], n=n[1], activation=activation)
        M0 = self.net_blocks.separable_conv_block(M0, c[1], 3, s[1], activation=None)
        A1 = add([M0, M1])
        
        M2 = self.net_blocks.inverted_residual_block(A1, c=c[2], ks=3, t=t[2], s=s[2], n=n[2], activation=activation)
        A1 = self.net_blocks.separable_conv_block(A1, c[2], 3, s[2], activation=None)
        A2 = add([A1, M2])
        
        M3 = self.net_blocks.inverted_residual_block(A2, c=c[3], ks=3, t=t[3], s=s[3], n=n[3], activation=activation)
        A2 = self.net_blocks.separable_conv_block(A2, c[3], 3, s[3], activation=None)
        A3 = add([A2, M3])
        
        M4 = self.net_blocks.inverted_residual_block(A3, c=c[4], ks=3, t=t[4], s=s[4], n=n[4], activation=activation)
        A3 = self.net_blocks.separable_conv_block(A3, c[4], 3, s[4], activation=None)
        A4 = add([A3, M4])
        
        M = self.net_blocks.spp_block(A4, pool_list=[1, 2, 4])
        
        self.backbone = Model(inputs=I, outputs=M, name=self.model_name)
    
    def build_mobilenet_v1_backbone(self, input_shape, alpha=1.0):
        
        I = Input(shape = input_shape)
        activation = 'relu'
        c = int(32 * alpha)

        x = self.net_blocks.conv_block(I, 32, 3, 2, activation=activation)

        x = self.net_blocks.bottleneck_v1(x, 64 , 3, s=1, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 128, 3, s=2, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 128, 3, s=1, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 256, 3, s=2, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 256, 3, s=1, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 512, 3, s=2, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 512, 3, s=1, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 512, 3, s=1, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 512, 3, s=1, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 512, 3, s=1, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 512, 3, s=1, alpha=alpha, activation=activation)

        x = self.net_blocks.bottleneck_v1(x, 1024, 3, s=2, alpha=alpha, activation=activation)
        x = self.net_blocks.bottleneck_v1(x, 1024, 3, s=1, alpha=alpha, activation=activation)

        x = GlobalAveragePooling2D()(x)

        self.backbone = Model(inputs=I, outputs=x, name=self.model_name)
    
    def build_mobilenet_v2_backbone(self, input_shape, alpha=1.0):
        
        c = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        t = [1, 1, 6, 6, 6, 6, 6, 6, 1]
        s = [2, 1, 2, 2, 2, 1, 2, 1, 1]
        n = [1, 1, 2, 3, 4, 3, 3, 1, 1]

        activation = 'relu6'
        I = Input(shape = input_shape)
        n_filters = self.net_blocks.make_divisible(c[0] * alpha, 8)

        x = self.net_blocks.conv_block(I, n_filters, 3, s[0], activation=activation) # (64, 64, 32) 

        x = self.net_blocks.inverted_residual_block(x, c=c[1], ks=3, t=t[1], s=s[1], n=n[1], alpha=alpha, activation=activation)
        x = self.net_blocks.inverted_residual_block(x, c=c[2], ks=3, t=t[2], s=s[2], n=n[2], alpha=alpha, activation=activation)
        x = self.net_blocks.inverted_residual_block(x, c=c[3], ks=3, t=t[3], s=s[3], n=n[3], alpha=alpha, activation=activation)
        x = self.net_blocks.inverted_residual_block(x, c=c[4], ks=3, t=t[4], s=s[4], n=n[4], alpha=alpha, activation=activation)
        x = self.net_blocks.inverted_residual_block(x, c=c[5], ks=3, t=t[5], s=s[5], n=n[5], alpha=alpha, activation=activation)
        x = self.net_blocks.inverted_residual_block(x, c=c[6], ks=3, t=t[6], s=s[6], n=n[6], alpha=alpha, activation=activation)
        x = self.net_blocks.inverted_residual_block(x, c=c[7], ks=3, t=t[7], s=s[7], n=n[7], alpha=alpha, activation=activation)

        if alpha > 1.0:
            last_filters = self.net_blocks.make_divisible(c[8] * alpha, 8)
        else:
            last_filters = c[8]
        x = self.net_blocks.conv_block(x, last_filters, 1, 1, activation=activation)

        x = GlobalAveragePooling2D()(x)

        self.backbone = Model(inputs=I, outputs=x, name=self.model_name)
    
    def build_mobilenet_v3_backbone(self, input_shape, alpha=1.0):
        
        I = Input(shape = input_shape)
    
        x = self.net_blocks.conv_block(I, 16, 3 , 2, activation='hard_swish')
        
        x = self.net_blocks.bottleneck_v3(x, 16 , 3, e=16 , s=1, alpha=alpha, squeeze=False, activation='relu6')

        x = self.net_blocks.bottleneck_v3(x, 24 , 3, e=64 , s=2, alpha=alpha, squeeze=False, activation='relu6')
        x = self.net_blocks.bottleneck_v3(x, 24 , 3, e=72 , s=1, alpha=alpha, squeeze=False, activation='relu6')

        x = self.net_blocks.bottleneck_v3(x, 40 , 5, e=72 , s=2, alpha=alpha, squeeze=True, activation='relu6')
        x = self.net_blocks.bottleneck_v3(x, 40 , 5, e=120, s=1, alpha=alpha, squeeze=True, activation='relu6')
        x = self.net_blocks.bottleneck_v3(x, 40 , 5, e=120, s=1, alpha=alpha, squeeze=True, activation='relu6')

        x = self.net_blocks.bottleneck_v3(x, 80 , 3, e=240, s=2, alpha=alpha, squeeze=False, activation='hard_swish')
        x = self.net_blocks.bottleneck_v3(x, 80 , 3, e=200, s=1, alpha=alpha, squeeze=False, activation='hard_swish')
        x = self.net_blocks.bottleneck_v3(x, 80 , 3, e=184, s=1, alpha=alpha, squeeze=False, activation='hard_swish')
        x = self.net_blocks.bottleneck_v3(x, 80 , 3, e=184, s=1, alpha=alpha, squeeze=False, activation='hard_swish')

        x = self.net_blocks.bottleneck_v3(x, 112, 3, e=480, s=1, alpha=alpha, squeeze=True, activation='hard_swish')
        x = self.net_blocks.bottleneck_v3(x, 112, 3, e=672, s=1, alpha=alpha, squeeze=True, activation='hard_swish')

        x = self.net_blocks.bottleneck_v3(x, 160, 5, e=672, s=2, alpha=alpha, squeeze=True, activation='hard_swish')
        x = self.net_blocks.bottleneck_v3(x, 160, 5, e=960, s=1, alpha=alpha, squeeze=True, activation='hard_swish')
        x = self.net_blocks.bottleneck_v3(x, 160, 5, e=960, s=1, alpha=alpha, squeeze=True, activation='hard_swish')

        x = self.net_blocks.conv_block(x, 960, 1, 1, activation='hard_swish')

        x = GlobalAveragePooling2D()(x)
        
        self.backbone = Model(inputs=I, outputs=x, name=self.model_name)
    
    def build_mobilefacenet_backbone(self, input_shape, alpha=1.0):
        
        c = [64, 64, 64, 128, 128, 128, 128]
        t = [1, 1, 2, 4, 2, 4, 2]
        s = [2, 1, 2, 2, 1, 2, 1]
        n = [1, 1, 5, 1, 6, 1, 2]
        activation='prelu'
        I = Input(shape = input_shape)

        x = self.net_blocks.conv_block(I, c[0], 3, s[0], activation=activation) 
        x = self.net_blocks.separable_conv_block(M, c[1], 3, s[1], activation=activation)

        x = self.net_blocks.inverted_residual_block(x, c=c[2], ks=3, t=t[2], s=s[2], n=n[2], activation=activation)
        x = self.net_blocks.inverted_residual_block(x, c=c[3], ks=3, t=t[3], s=s[3], n=n[3], activation=activation)
        x = self.net_blocks.inverted_residual_block(x, c=c[4], ks=3, t=t[4], s=s[4], n=n[4], activation=activation)
        x = self.net_blocks.inverted_residual_block(x, c=c[5], ks=3, t=t[5], s=s[5], n=n[5], activation=activation)
        x = self.net_blocks.inverted_residual_block(x, c=c[6], ks=3, t=t[6], s=s[6], n=n[6], activation=activation)

        x = self.net_blocks.conv_block(x, 512, 1, 1, 'valid', activation=activation)
        ks = K.int_shape(x)[2]
        x = self.net_blocks.depthwise_conv_block(x, ks, 1, padding='valid', activation=None)
        
        self.backbone = Model(inputs=I, outputs=x, name=self.model_name)
        
    def build_softmax_model(self, n_classes):
        
        I=self.backbone.inputs
        x=self.backbone.outputs[0]
        
        if(len(x.shape)==2):
            c = K.int_shape(x)[self.net_blocks.channel_axis]
            x = Reshape((1, 1, c))(x)
        x = self.net_blocks.conv_block(x, self.embedding_dim, 1, 1, 'valid', activation=None) 
        
        if(self.dropout>0):
            x = Dropout(rate=dropout)(x)
    
        x = self.net_blocks.conv_block(x, n_classes, 1, 1, activation='softmax', norm=None)
        x = Reshape((n_classes,))(x)
        
        self.softmax_model = Model(inputs=I, outputs=x, name=self.model_name) 

    def build_adacos_model(self):
        
        label = Input(shape=(1,), name='label_input')
        softmax = self.softmax_model.outputs[0]
        n_classes = K.int_shape(softmax)[-1]
        inputs = self.softmax_model.inputs[0]
        x = self.softmax_model.layers[self.end_fine_tune_layer_id].output
        
        if(self.dropout>0):
            x = Dropout(rate=dropout)(x)
                
        x = Flatten(name=self.embedding_layer_name)(x)
            
        break_point = len(self.softmax_model.layers) + self.start_fine_tune_layer_id
        
        for layer in self.softmax_model.layers[:break_point]:
            layer.trainable=False
        
        outputs = AdaCos(n_classes, initializer=self.net_blocks.kernel_initializer, regularizer=self.net_blocks.kernel_regularizer, name='adacos')([x, label])
        
        self.adacos_model = Model(inputs = (inputs, label), outputs = outputs, name=self.model_name)

