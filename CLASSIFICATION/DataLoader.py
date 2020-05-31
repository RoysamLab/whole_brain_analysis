import os
import h5py
import scipy
import random
import numpy as np
import pandas as pd
from prepare_data.utils import center_image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BrainLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.augment = cfg.data_augment
        self.max_angle = cfg.max_angle
        self.batch_size = cfg.batch_size
        self.data_path = os.path.join(cfg.OUTPUT_DIR, 'data.h5')

    def get_data(self, mode='train'):
        h5f = h5py.File(self.data_path, 'r')
        if mode == 'train':
            x_train = h5f['X_train'][:]
            y_train = h5f['Y_train'][:]
            self.x_train, self.y_train = self.preprocess(x_train, y_train)
        elif mode == 'valid':
            x_valid = h5f['X_test'][:]
            y_valid = h5f['Y_test'][:]
            self.x_valid, self.y_valid = self.preprocess(x_valid, y_valid)
        elif mode == 'test':
            x_test = h5f['X_test'][:]
            y_test = h5f['Y_test'][:]
            self.x_test, self.y_test = self.preprocess(x_test, y_test)
            bbxs = h5f['bbxs'][:]
            self.bbxs = pd.DataFrame(bbxs, columns=['ID', 'centroid_x', 'centroid_y', 'xmin', 'ymin', 'xmax', 'ymax'])
            self.bbxs.set_index('ID', inplace=True)
            self.image_size = tuple(h5f['image_size'][:])
            self.biomarkers = h5f['biomarkers'][:].astype(str)
        h5f.close()

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            x = self.x_train[start:end]
            y = self.y_train[start:end]
            if self.augment:
                x = random_rotation_2d(x, self.cfg.max_angle)
        elif mode == 'valid':
            x = self.x_valid[start:end]
            y = self.y_valid[start:end]
        elif mode == 'test':
            x = self.x_test[start:end]
            y = self.y_test[start:end]
        return x, y

    def count_num_batch(self, batch_size, mode='train'):
        if mode == 'train':
            num_batch = int(self.y_train.shape[0] / batch_size)
        elif mode == 'valid':
            num_batch = int(self.y_valid.shape[0] / batch_size)
        elif mode == 'test':
            num_batch = int(np.ceil(self.y_test.shape[0] / batch_size))
        return num_batch

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        self.x_train = self.x_train[permutation, :, :, :]
        self.y_train = self.y_train[permutation]

    def preprocess(self, x, y, normalize=None, one_hot=False):
        if normalize == 'standard':
            self.get_stats()
            x = (x - self.mean) / self.std
        elif normalize == 'unity_based':
            x /= 65535.
        x = x.reshape((-1, self.cfg.height, self.cfg.width, self.cfg.channel)).astype(np.float32)
        if one_hot:
            y = (np.arange(self.cfg.num_cls) == y[:, None]).astype(np.float32)
        return x, y

    def get_stats(self):
        h5f = h5py.File(self.data_path, 'r')
        x_train = h5f['X_train'][:]
        h5f.close()
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)

    def plot_hists(self, y_prob):
        fig, ax = plt.subplots(y_prob.shape[1], figsize=(8, 8))
        plt.subplots_adjust(hspace=1)
        for i in range(y_prob.shape[1]):
            ax[i].hist(y_prob[:, i], bins=10000, color='k')
            ax[i].set_title(self.biomarkers[i + 2])
            ax[i].xaxis.set_major_locator(plt.MultipleLocator(0.1))
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        plt.savefig(os.path.join(self.cfg.OUTPUT_DIR, 'histograms.png'), dpi=1000)

    def generate_center_images(self, y_pred):
        centers = self.bbxs[['centroid_x', 'centroid_y']].values
        neg_samples = np.where(~y_pred.any(axis=1))[0]
        center_image(os.path.join(self.cfg.OUTPUT_DIR, 'all.tif'), centers, self.image_size)
        center_image(os.path.join(self.cfg.OUTPUT_DIR, 'uncategorized.tif'), centers[neg_samples, :], self.image_size)
        for i in range(y_pred.shape[1]):
            center_image(os.path.join(self.cfg.OUTPUT_DIR, self.biomarkers[i + 2] + '.tif'),
                         centers[y_pred[:, i] == 1, :], self.image_size)

    def generate_classification_table(self, y_pred):
        # TODO: save as int
        for i in range(y_pred.shape[1]):
            self.bbxs[self.biomarkers[i + 2]] = y_pred[:, i]

        self.bbxs.to_csv(os.path.join(self.cfg.OUTPUT_DIR, 'classification_table.csv'))

    def generate_probability_table(self, y_prob):
        for i in range(y_prob.shape[1]):
            self.bbxs[self.biomarkers[i + 2]] = y_prob[:, i]

        self.bbxs.to_csv(os.path.join(self.cfg.OUTPUT_DIR, 'probability_table.csv'))


def random_rotation_2d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).
    Arguments:
    max_angle: `float`. The maximum rotation angle.
    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image = np.squeeze(batch[i])
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)
