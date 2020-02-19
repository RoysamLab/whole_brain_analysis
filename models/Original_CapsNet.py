from models.base_model import BaseModel
from layers.FC_Caps import FCCapsuleLayer
from keras import layers
import tensorflow as tf

from layers.ops import squash


class Orig_CapsNet(BaseModel):
    def __init__(self, sess, conf):
        super(Orig_CapsNet, self).__init__(sess, conf)
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            # Layer 1: A 2D conv layer
            conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1,
                                  padding='valid', activation='relu', name='conv1')(x)

            # Layer 2: Primary Capsule Layer; simply a 2D conv + reshaping
            primary_caps = layers.Conv2D(filters=256, kernel_size=9, strides=2,
                                         padding='valid', activation='relu', name='primary_caps')(conv1)
            _, H, W, dim = primary_caps.get_shape()
            num_caps = int(H.value * W.value * dim.value / self.conf.prim_caps_dim)
            primary_caps_reshaped = layers.Reshape((num_caps, self.conf.prim_caps_dim))(primary_caps)
            caps1_output = squash(primary_caps_reshaped)

            # Layer 3: Digit Capsule Layer; Here is where the routing takes place
            self.digit_caps = FCCapsuleLayer(num_caps=self.conf.num_cls, caps_dim=self.conf.digit_caps_dim,
                                             routings=3, name='digit_caps')(caps1_output)
            # [?, 10, 16]

            self.mask()
            self.decoder()

    def decoder(self):
        with tf.variable_scope('Decoder'):
            decoder_input = tf.reshape(self.output_masked, [-1, self.conf.num_cls * self.conf.digit_caps_dim])
            # [?, 160]
            fc1 = tf.layers.dense(decoder_input, self.conf.h1, activation=tf.nn.relu, name="FC1")
            # [?, 512]
            fc2 = tf.layers.dense(fc1, self.conf.h2, activation=tf.nn.relu, name="FC2")
            # [?, 1024]
            self.decoder_output = tf.layers.dense(fc2, self.conf.width * self.conf.height * self.conf.channel,
                                                  activation=tf.nn.sigmoid, name="FC3")
            # [?, 784]
