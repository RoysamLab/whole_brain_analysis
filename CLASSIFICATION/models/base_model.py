import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from models.utils.loss_ops import margin_loss, spread_loss


class BaseModel(object):
    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.summary_list = []
        if self.conf.dim == 2:
            self.input_shape = [None, self.conf.height, self.conf.width, self.conf.channel]
        else:
            self.input_shape = [None, self.conf.height, self.conf.width, self.conf.channel]

        self.output_shape = [None, self.conf.num_cls]
        self.create_placeholders()
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                           trainable=False)

    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.float32, self.output_shape, name='annotation')
            self.mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

    def mask(self):
        with tf.variable_scope('Masking'):
            epsilon = 1e-9
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=2, keepdims=True) + epsilon)
            # [?, num_cls, 1]

            self.y_prob = tf.squeeze(self.v_length)     # [?, num_cls]

            y_prob_argmax = tf.to_int32(tf.argmax(self.v_length, axis=1))
            # [?, 1]
            self.y_pred = tf.squeeze(y_prob_argmax)
            # [?] (predicted labels)
            self.y_pred_ohe = tf.one_hot(self.y_pred, depth=self.conf.num_cls)
            # [?, 10] (one-hot-encoded predicted labels)

            reconst_targets = tf.cond(self.mask_with_labels,  # condition
                                      lambda: self.y,  # if True (Training)
                                      lambda: self.y_pred_ohe,  # if False (Test)
                                      name="reconstruction_targets")
            # [?, 10]
            self.output_masked = tf.multiply(self.digit_caps, tf.expand_dims(reconst_targets, -1))
            # [?, 10, 16]

    def loss_func(self):
        with tf.variable_scope('Loss'):
            if self.conf.loss_type == 'margin':
                loss = margin_loss(self.y, self.v_length, self.conf)
                self.summary_list.append(tf.summary.scalar('margin', loss))
            elif self.conf.loss_type == 'spread':
                self.generate_margin()
                loss = spread_loss(self.y, self.act, self.margin, 'spread_loss')
                self.summary_list.append(tf.summary.scalar('spread_loss', loss))
            if self.conf.L2_reg:
                with tf.name_scope('l2_loss'):
                    l2_loss = tf.reduce_sum(self.conf.lmbda * tf.stack([tf.nn.l2_loss(v)
                                                                        for v in tf.get_collection('weights')]))
                    loss += l2_loss
                self.summary_list.append(tf.summary.scalar('l2_loss', l2_loss))
            if self.conf.add_recon_loss or self.conf.add_decoder:
                with tf.variable_scope('Reconstruction_Loss'):
                    orgin = tf.reshape(self.x, shape=(-1, self.conf.height * self.conf.width * self.conf.channel))
                    squared = tf.square(self.decoder_output - orgin)
                    self.recon_err = tf.reduce_mean(squared)
                    self.total_loss = loss + self.conf.alpha * self.conf.width * self.conf.height * self.recon_err
                    self.summary_list.append(tf.summary.scalar('reconstruction_loss', self.recon_err))
                    recon_img = tf.reshape(self.decoder_output,
                                           shape=(-1, self.conf.height, self.conf.width, self.conf.channel))
            else:
                self.total_loss = loss
            self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

        if self.conf.add_recon_loss or self.conf.add_decoder:
            # self.summary_list.append(tf.summary.image('reconstructed', recon_img))
            # self.summary_list.append(tf.summary.image('original', self.x))
            pass

    def accuracy_func(self):
        with tf.variable_scope('Accuracy'):
            correct_prediction = tf.equal(tf.to_int32(tf.argmax(self.y, axis=1)), self.y_pred)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def generate_margin(self):
        # margin schedule
        # margin increase from 0.2 to 0.9 after margin_schedule_epoch_achieve_max
        NUM_STEPS_PER_EPOCH = int(self.conf.N / self.conf.batch_size)
        margin_schedule_epoch_achieve_max = 10.0
        self.margin = tf.train.piecewise_constant(tf.cast(self.global_step, dtype=tf.int32),
                                                  boundaries=[int(NUM_STEPS_PER_EPOCH *
                                                                  margin_schedule_epoch_achieve_max * x / 7)
                                                              for x in xrange(1, 8)],
                                                  values=[x / 10.0 for x in range(2, 10)])

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()

        with tf.name_scope('Optimizer'):
            with tf.name_scope('Learning_rate_decay'):
                learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                           self.global_step,
                                                           decay_steps=3000,
                                                           decay_rate=0.97,
                                                           staircase=True)
                self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
            self.summary_list.append(tf.summary.scalar('learning_rate', self.learning_rate))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            """Compute gradient."""
            grads = optimizer.compute_gradients(self.total_loss)
            # grad_check = [tf.check_numerics(g, message='Gradient NaN Found!') for g, _ in grads if g is not None] \
            #              + [tf.check_numerics(self.total_loss, message='Loss NaN Found')]
            """Apply graident."""
            # with tf.control_dependencies(grad_check):
            #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #     with tf.control_dependencies(update_ops):
            """Add graident summary"""
            # for grad, var in grads:
            #     self.summary_list.append(tf.summary.histogram(var.name, grad))
            if self.conf.grad_clip:
                """Clip graident."""
                grads = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grads]
            """NaN to zero graident."""
            # grads = [(tf.where(tf.is_nan(grad), tf.zeros(grad.shape), grad), var) for grad, var in grads]
            self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
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
        summary_list = [tf.summary.scalar('Loss/total_loss', self.mean_loss),
                        tf.summary.scalar('Accuracy/average_accuracy', self.mean_accuracy)] + self.summary_list
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step, mode):
        # print('----> Summarizing at step {}'.format(step))
        if mode == 'train':
            self.train_writer.add_summary(summary, step)
        elif mode == 'valid':
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        self.best_validation_accuracy = 0

        from DataLoader import BrainLoader as DataLoader
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='train')
        self.data_reader.get_data(mode='valid')
        self.train_loop()

    def train_loop(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('*' * 50)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
            print('*' * 50)
        else:
            print('*' * 50)
            print('----> Start Training')
            print('*' * 50)
        self.num_val_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='valid')
        if self.conf.epoch_based:
            self.num_train_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='train')
            for epoch in range(self.conf.max_epoch):
                self.data_reader.randomize()
                for train_step in range(self.num_train_batch):
                    glob_step = epoch * self.num_train_batch + train_step
                    start = train_step * self.conf.batch_size
                    end = (train_step + 1) * self.conf.batch_size
                    x_batch, y_batch = self.data_reader.next_batch(start, end, mode='train')
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.mask_with_labels: True}
                    if train_step % self.conf.SUMMARY_FREQ == 0:
                        _, _, _, summary = self.sess.run([self.train_op,
                                                          self.mean_loss_op,
                                                          self.mean_accuracy_op,
                                                          self.merged_summary], feed_dict=feed_dict)
                        loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                        self.save_summary(summary, glob_step + self.conf.reload_step, mode='train')
                        print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                    else:
                        self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
                self.evaluate(glob_step)
        else:
            self.data_reader.randomize()
            for train_step in range(1, self.conf.max_step + 1):
                # print(train_step)
                self.is_train = True
                if train_step % self.conf.SUMMARY_FREQ == 0:
                    x_batch, y_batch = self.data_reader.next_batch()
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.mask_with_labels: True}
                    _, _, _, summary = self.sess.run([self.train_op,
                                                      self.mean_loss_op,
                                                      self.mean_accuracy_op,
                                                      self.merged_summary], feed_dict=feed_dict)
                    loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                    self.save_summary(summary, train_step + self.conf.reload_step, mode='train')
                    print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step, loss, acc))
                else:
                    x_batch, y_batch = self.data_reader.next_batch()
                    feed_dict = {self.x: x_batch, self.y: y_batch, self.mask_with_labels: True}
                    self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
                if train_step % self.conf.VAL_FREQ == 0:
                    self.evaluate(train_step)

    def evaluate(self, train_step):
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_val_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
            feed_dict = {self.x: x_val, self.y: y_val, self.mask_with_labels: False}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)

        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, train_step + self.conf.reload_step, mode='valid')
        if valid_acc > self.best_validation_accuracy:
            self.best_validation_accuracy = valid_acc
            improved_str = '(improved)'
            self.save(train_step + self.conf.reload_step)
        else:
            improved_str = ''
        print('-' * 25 + 'Validation' + '-' * 25)
        print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}{3}'
              .format(train_step, valid_loss, valid_acc, improved_str))
        print('-' * 60)

    def test(self, step_num):
        self.sess.run(tf.local_variables_initializer())
        self.reload(step_num)

        from DataLoader import BrainLoader as DataLoader
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='test')
        self.num_test_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='test')
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_test_batch):
            start = step * self.conf.batch_size
            end = (step + 1) * self.conf.batch_size
            x_test, y_test = self.data_reader.next_batch(start, end, mode='test')
            feed_dict = {self.x: x_test, self.y: y_test, self.mask_with_labels: False}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
        test_loss, test_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        print('-' * 18 + 'Test Completed' + '-' * 18)
        print('test_loss= {0:.4f}, test_acc={1:.01%}'.format(test_loss, test_acc))
        print('-' * 50)

    def inference(self, step_num):
        self.sess.run(tf.local_variables_initializer())
        self.reload(step_num)

        from DataLoader import BrainLoader as DataLoader
        self.data_reader = DataLoader(self.conf)
        self.data_reader.get_data(mode='test')
        self.num_test_batch = self.data_reader.count_num_batch(self.conf.batch_size, mode='test')
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())

        if self.conf.test_mode == 'first':
            y_prob = []
            y_pred = []
            for step in tqdm(range(self.num_test_batch)):
                start = step * self.conf.batch_size
                end = (step + 1) * self.conf.batch_size
                x_test, _ = self.data_reader.next_batch(start, end, mode='test')
                feed_dict = {self.x: x_test, self.mask_with_labels: False}
                y_prob.extend(self.sess.run(self.y_prob, feed_dict=feed_dict))
                y_pred.extend(self.sess.run(self.y_pred_ohe, feed_dict=feed_dict))
            y_prob = np.array(y_prob)
            y_pred = np.array(y_pred)

            # generate probability table
            self.data_reader.generate_probability_table(y_prob)

            with h5py.File(os.path.join(self.conf.OUTPUT_DIR, 'y.h5'), 'w') as f:
                f.create_dataset('y_pred', data=y_pred)
                f.create_dataset('y_prob', data=y_prob)

            # # generate histograms
            # self.data_reader.plot_hists(y_prob)

            # set threshold of 0.5 for each class to threshold probabilites
            thresh = np.ones(self.conf.num_cls) * .5

        # if it is not the first time for inference, load probabilities from y.h5 and adjust the thresholds
        elif self.conf.test_mode == 'adjust':
            with h5py.File(os.path.join(self.conf.OUTPUT_DIR, 'y.h5'), 'r') as f:
                y_prob = f['y_prob'][:]
                y_pred = f['y_pred'][:]

            # read thresholds from user
            thresh = self.conf.thresholds

        # create negative samples
        # TODO: fix for any number of class
        neg_samples = np.where((y_prob[:, 0] < thresh[0]) &
                               (y_prob[:, 1] < thresh[1]) &
                               (y_prob[:, 2] < thresh[2]) &
                               (y_prob[:, 3] < thresh[3]) &
                               (y_prob[:, 4] < thresh[4]))[0]
        print('precentage of uncategorized cells: {:.2%}'.format((len(neg_samples) / len(self.data_reader.bbxs))))
        y_pred[neg_samples, :] = np.zeros((y_pred.shape[1]))

        # generate classification and proability table
        self.data_reader.generate_classification_table(y_pred)

        # center image
        self.data_reader.generate_center_images(y_pred)

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')
