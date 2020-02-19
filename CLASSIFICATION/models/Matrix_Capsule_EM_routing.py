from base_model import BaseModel
import tensorflow as tf
from models.utils.ops import conv_2d, capsules_init, capsule_conv, capsule_fc


class MatrixCapsNet(BaseModel):
    def __init__(self, sess, conf):
        super(MatrixCapsNet, self).__init__(sess, conf)
        self.is_train = True
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            net, summary = conv_2d(x, 5, 2, self.conf.A, 'CONV1', add_bias=self.conf.use_bias,
                                   add_reg=self.conf.L2_reg, batch_norm=self.conf.use_BN, is_train=self.is_train)
            # [?, 14, 14, A]
            self.summary_list.append(summary)

            pose, act, summary_list = capsules_init(net, 1, 1, OUT=self.conf.B, padding='VALID',
                                                    pose_shape=[4, 4], add_reg=self.conf.L2_reg,
                                                    use_bias=self.conf.use_bias, name='capsule_init')
            # [?, 14, 14, B, 4, 4], [?, 14, 14, B]
            for summary in summary_list:
                self.summary_list.append(summary)

            pose, act, summary_list = capsule_conv(pose, act, K=3, OUT=self.conf.C, stride=2, add_reg=self.conf.L2_reg,
                                                   iters=self.conf.iter, std=1, name='capsule_conv1')
            # [?, 6, 6, C, 4, 4], [?, 6, 6, C]
            for summary in summary_list:
                self.summary_list.append(summary)

            pose, act, summary_list = capsule_conv(pose, act, K=3, OUT=self.conf.D, stride=1, add_reg=self.conf.L2_reg,
                                                   iters=self.conf.iter, std=1, name='capsule_conv2')
            # [?, 4, 4, D, 4, 4], [?, 4, 4, D]
            for summary in summary_list:
                self.summary_list.append(summary)

            self.pose, self.act, summary_list = capsule_fc(pose, act, OUT=self.conf.num_cls, add_reg=self.conf.L2_reg,
                                                           iters=self.conf.iter, std=1, add_coord=self.conf.add_coords,
                                                           name='capsule_fc')
            # [?, num_cls, 4, 4], [?, num_cls]
            for summary in summary_list:
                self.summary_list.append(summary)

            self.v_length = self.act

            self.y_pred = tf.to_int32(tf.argmax(self.act, axis=1))
            if self.conf.add_decoder:
                self.decoder()

    def decoder(self):
        with tf.variable_scope('Decoder'):
            decoder_input = tf.reshape(self.pose, [-1, self.conf.num_cls * self.conf.digit_caps_dim])
            # [?, 160]
            fc1 = tf.layers.dense(decoder_input, self.conf.h1, activation=tf.nn.relu, name="FC1")
            # [?, 512]
            fc2 = tf.layers.dense(fc1, self.conf.h2, activation=tf.nn.relu, name="FC2")
            # [?, 1024]
            self.decoder_output = tf.layers.dense(fc2, self.conf.width * self.conf.height * self.conf.channel,
                                                  activation=tf.nn.sigmoid, name="FC3")
            # [?, 784]
