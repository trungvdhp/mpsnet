import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import Constant
from tensorflow.python.framework import tensor_util as tf_utils

def resolve_training(layer, training):
    if training is None:
        training = K.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        # When the layer is not trainable, override the value
        training = False
    return tf_utils.constant_value(training)

class AdaCos(Layer):
    
    def __init__(self, 
                 n_classes,
                 is_dynamic=True,
                 initializer=None,
                 regularizer=None,
                 **kwargs):
        
        super(AdaCos, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.is_dynamic = is_dynamic
        self.initializer = initializer
        self.regularizer = regularizer
        self.init_s = math.sqrt(2) * math.log(n_classes - 1)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            'is_dynamic': self.is_dynamic,
            'regularizer': self.regularizer
        })
        return config
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(AdaCos, self).build(input_shape[0])
        
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1].value, self.n_classes),
                                 initializer=self.initializer,
                                 trainable=True,
                                 regularizer=self.regularizer)
        if self.is_dynamic:
            self.s = self.add_weight(shape=(),
                                     initializer=Constant(self.init_s),
                                     trainable=False,
                                     aggregation=tf.VariableAggregation.MEAN)

    def call(self, inputs, training=None):
        x, y = inputs
        # Squeezing is necessary for Keras, it remove dimension 1, (None, 1) => (None,)
        y = tf.reshape(y, [-1])
        # Convert label to int32 as input type for onehot function
        y = tf.cast(y, tf.int32)
        
        # Normalize features and weights and compute dot product
        x = tf.nn.l2_normalize(x, axis=1, name='normed_embeddings')
        w = tf.nn.l2_normalize(self.W, axis=0, name='normed_weights')
        logits = tf.matmul(x, w, name='cos_t')
        
        # Fixed AdaCos
        is_dynamic = tf_utils.constant_value(self.is_dynamic)
        
        if not is_dynamic:
            # s is not created since we are not in dynamic mode
            logits = self.init_s * logits
        else:
            training = resolve_training(self, training)

            if not training:
                # We don't have labels to update s if we're not in training mode
                logits = self.s * logits
            else:
                theta = tf.math.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
                one_hot = tf.one_hot(y, depth=self.n_classes)
                b_avg = tf.where(one_hot < 1.0,
                                 tf.exp(self.s * logits),
                                 tf.zeros_like(logits))
                b_avg = tf.reduce_mean(tf.reduce_sum(b_avg, axis=1), name='B_avg')
                #tf.squeeze(y)
                theta_class = tf.gather_nd(theta, 
                                           tf.stack([tf.range(tf.shape(y)[0]), y], axis=1),
                                           name='theta_class')
                mid_index = tf.shape(theta_class)[0] // 2 + 1
                theta_med = tf.nn.top_k(theta_class, mid_index).values[-1]

                self.s.assign(tf.math.log(b_avg) / tf.math.cos(tf.minimum(math.pi/4, theta_med)))
                # Scaled logits
                logits = self.s * logits
            
        return tf.nn.softmax(logits)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_classes)