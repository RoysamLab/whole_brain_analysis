from layers.FC_Caps import FCCapsuleLayer
from keras import layers
from models.utils.ops import *
from layers.ops import squash
import numpy as np


class Fast_CapsNet_3D:
    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.input_shape = [None, self.conf.height, self.conf.width, self.conf.depth, self.conf.channel]
        self.output_shape = [None, self.conf.num_cls]
        self.create_placeholders()
        self.build_network(self.x)
        self.configure_network()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.float32, self.output_shape, name='annotation')
            self.mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            # Layer 1: A 3D conv layer
            conv1 = layers.Conv3D(filters=256, kernel_size=9, strides=1,
                                  padding='valid', activation='relu', name='conv1')(x)

            # Layer 2: Primary Capsule Layer; simply a 3D conv + reshaping
            primary_caps = layers.Conv3D(filters=256, kernel_size=9, strides=2,
                                         padding='valid', activation='relu', name='primary_caps')(conv1)
            _, H, W, D, dim = primary_caps.get_shape()
            primary_caps_reshaped = layers.Reshape((H.value * W.value * D.value, dim.value))(primary_caps)
            caps1_output = squash(primary_caps_reshaped)
            # [?, 512, 256]
            # Layer 3: Digit Capsule Layer; Here is where the routing takes place
            digitcaps_layer = FCCapsuleLayer(num_caps=self.conf.num_cls, caps_dim=self.conf.digit_caps_dim,
                                             routings=3, name='digit_caps')
            self.digit_caps = digitcaps_layer(caps1_output)  # [?, 2, 16]
            u_hat = digitcaps_layer.get_predictions(caps1_output)  # [?, 2, 512, 16]
            u_hat_shape = u_hat.get_shape().as_list()
            self.img_s = int(round(u_hat_shape[2] ** (1. / 3)))
            self.u_hat = layers.Reshape(
                (self.conf.num_cls, self.img_s, self.img_s, self.img_s, 1, self.conf.digit_caps_dim))(u_hat)
            # self.u_hat = tf.transpose(u_hat, perm=[1, 0, 2, 3, 4, 5, 6])
            # u_hat: [?, 2, 8, 8, 8, 1, 16]
            self.decoder()

    def decoder(self):
        with tf.variable_scope('Decoder'):
            epsilon = 1e-9
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=2, keep_dims=True) + epsilon)
            # [batch_size, 2, 1]
            indices = tf.argmax(tf.squeeze(self.v_length), axis=1)
            self.y_prob = tf.nn.softmax(tf.squeeze(self.v_length))
            self.y_pred = tf.squeeze(tf.to_int32(tf.argmax(self.v_length, axis=1)))
            # [batch_size] (predicted labels)
            y_pred_ohe = tf.one_hot(self.y_pred, depth=self.conf.num_cls)
            # [batch_size, 2] (one-hot-encoded predicted labels)

            reconst_targets = tf.cond(self.mask_with_labels,  # condition
                                      lambda: self.y,  # if True (Training)
                                      lambda: y_pred_ohe,  # if False (Test)
                                      name="reconstruction_targets")
            # [batch_size, 2]
            reconst_targets = tf.reshape(reconst_targets, (-1, 1, 1, 1, self.conf.num_cls))
            # [batch_size, 1, 1, 1, 2]
            reconst_targets = tf.tile(reconst_targets, (1, self.img_s, self.img_s, self.img_s, 1))
            # [batch_size, 8, 8, 8, 2]

            num_partitions = self.conf.batch_size
            partitions = tf.range(num_partitions)
            u_list = tf.dynamic_partition(self.u_hat, partitions, num_partitions, name='uhat_dynamic_unstack')
            ind_list = tf.dynamic_partition(indices, partitions, num_partitions, name='ind_dynamic_unstack')

            a = tf.stack([tf.gather_nd(tf.squeeze(mat, axis=0), [[ind]]) for mat, ind in zip(u_list, ind_list)])
            # [?, 1, 8, 8, 8, 1, 16]
            feat = tf.reshape(a, (-1, self.img_s, self.img_s, self.img_s, self.conf.digit_caps_dim))
            # [?, 8, 8, 8, 16]
            self.cube = tf.concat([feat, reconst_targets], axis=-1)
            # [?, 8, 8, 8, 18]

            res1 = Deconv3D(self.cube,
                            filter_size=4,
                            num_filters=16,
                            stride=2,
                            layer_name="deconv_1",
                            out_shape=[self.conf.batch_size, 16, 16, 16, 16])
            self.decoder_output = Deconv3D(res1,
                                           filter_size=4,
                                           num_filters=1,
                                           stride=2,
                                           layer_name="deconv_2",
                                           out_shape=[self.conf.batch_size, 32, 32, 32, 1])

    def mask(self):
        with tf.variable_scope('Masking'):
            epsilon = 1e-9
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=2, keep_dims=True) + epsilon)
            # [?, 10, 1]

            y_prob_argmax = tf.to_int32(tf.argmax(self.v_length, axis=1))
            # [?, 1]
            self.y_pred = tf.squeeze(y_prob_argmax)
            # [?] (predicted labels)
            y_pred_ohe = tf.one_hot(self.y_pred, depth=self.conf.num_cls)
            # [?, 10] (one-hot-encoded predicted labels)

            reconst_targets = tf.cond(self.mask_with_labels,  # condition
                                      lambda: self.y,  # if True (Training)
                                      lambda: y_pred_ohe,  # if False (Test)
                                      name="reconstruction_targets")
            # [?, 10]

            self.output_masked = tf.multiply(self.digit_caps, tf.expand_dims(reconst_targets, -1))
            # [?, 10, 16]

    def loss_func(self):
        # 1. The margin loss
        with tf.variable_scope('Margin_Loss'):
            # max(0, m_plus-||v_c||)^2
            present_error = tf.square(tf.maximum(0., self.conf.m_plus - self.v_length))
            # [?, 10, 1]

            # max(0, ||v_c||-m_minus)^2
            absent_error = tf.square(tf.maximum(0., self.v_length - self.conf.m_minus))
            # [?, 10, 1]

            # reshape: [?, 10, 1] => [?, 10]
            present_error = tf.squeeze(present_error)
            absent_error = tf.squeeze(absent_error)

            T_c = self.y
            # [?, 10]
            L_c = T_c * present_error + self.conf.lambda_val * (1 - T_c) * absent_error
            # [?, 10]
            self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1), name="margin_loss")

        # 2. The reconstruction loss
        with tf.variable_scope('Reconstruction_Loss'):
            orgin = tf.reshape(self.x, shape=(-1, self.conf.height * self.conf.width * self.conf.depth))
            squared = tf.square(self.decoder_output - orgin)
            self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        with tf.variable_scope('Total_Loss'):
            self.total_loss = self.margin_loss + self.conf.alpha * self.reconstruction_err
            self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

    def accuracy_func(self):
        with tf.variable_scope('Accuracy'):
            correct_prediction = tf.equal(tf.to_int32(tf.argmax(self.y, axis=1)), self.y_pred)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        with tf.name_scope('Optimizer'):
            with tf.name_scope('Learning_rate_decay'):
                global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                              trainable=False)
                steps_per_epoch = self.conf.num_tr // self.conf.batch_size
                learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                           global_step,
                                                           steps_per_epoch,
                                                           0.97,
                                                           staircase=True)
                self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        recon_img = tf.reshape(self.decoder_output[:, :, :, 16, :],
                               shape=(-1, self.conf.height, self.conf.width, self.conf.channel))
        orig_image = tf.reshape(self.x[:, :, :, 16, :],
                                shape=(-1, self.conf.height, self.conf.width, self.conf.channel))
        summary_list = [tf.summary.scalar('Loss/total_loss', self.mean_loss),
                        tf.summary.scalar('Accuracy/average_accuracy', self.mean_accuracy),
                        tf.summary.image('original', orig_image),
                        tf.summary.image('reconstructed', recon_img)]
        self.merged_summary = tf.summary.merge(summary_list)

    def train(self):
        self.sess.run(tf.local_variables_initializer())

        from DataLoader import BrainLoader as DataLoader
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_validation()
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('*' * 50)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
            print('*' * 50)
        else:
            print('*' * 50)
            print('----> Start Training')
            print('*' * 50)
        self.num_train_batch = int(self.data_reader.y_train.shape[0] / self.conf.batch_size)
        self.num_val_batch = int(self.data_reader.y_valid.shape[0] / self.conf.val_batch_size)
        for epoch in range(self.conf.max_epoch):
            self.data_reader.randomize()
            for train_step in range(self.num_train_batch):
                start = train_step * self.conf.batch_size
                end = (train_step + 1) * self.conf.batch_size
                global_step = epoch * self.num_train_batch + train_step
                x_batch, y_batch = self.data_reader.next_batch(start, end)
                feed_dict = {self.x: x_batch, self.y: y_batch, self.mask_with_labels: True}
                if train_step % self.conf.SUMMARY_FREQ == 0:
                    _, _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.mean_accuracy_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                    self.save_summary(summary, global_step, mode='train')
                    print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                else:
                    self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)

            self.evaluate(epoch)
            self.save(epoch)

    def evaluate(self, epoch, global_step):
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_val_batch):
            start = step * self.conf.val_batch_size
            end = (step + 1) * self.conf.val_batch_size
            x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
            feed_dict = {self.x: x_val, self.y: y_val, self.mask_with_labels: False}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)

        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, global_step, mode='valid')
        print('-' * 30 + 'Validation' + '-' * 30)
        print('Epoch #{0} : val_loss= {1:.4f}, val_acc={2:.01%}'.format(epoch, valid_loss, valid_acc))
        print('-' * 70)