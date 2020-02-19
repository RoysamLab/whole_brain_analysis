import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tensorflow.examples.tutorials.mnist import input_data


def load_data(mode='train'):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if mode == 'train':
        x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, \
                                             mnist.validation.images, mnist.validation.labels
        x_train, _ = reformat(x_train, y_train)
        x_valid, _ = reformat(x_valid, y_valid)
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, y_test = mnist.test.images, mnist.test.labels
        x_test, _ = reformat(x_test, y_test)
    return x_test, y_test


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :, :, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def reformat(x, y):
    """
    Reformats the data to the format acceptable for convolutional layers
    :param x: input array
    :param y: corresponding labels
    :return: reshaped input and labels
    """
    img_size, num_ch, num_class = int(np.sqrt(x.shape[-1])), 1, len(np.unique(np.argmax(y, 1)))
    dataset = x.reshape((-1, img_size, img_size, num_ch)).astype(np.float32)
    labels = (np.arange(num_class) == y[:, None]).astype(np.float32)
    return dataset, labels


def write_spec(args):
    config_file = open(args.modeldir + args.run_name + '/config.txt', 'w')
    config_file.write('run_name: ' + args.run_name + '\n')
    config_file.write('model: ' + args.model + '\n')
    config_file.write('loss_type: ' + args.loss_type + '\n')
    config_file.write('add_recon_loss: ' + str(args.add_recon_loss) + '\n')
    config_file.write('data_path: ' + args.OUTPUT_DIR + '\n')
    config_file.write('height: ' + str(args.height) + '\n')
    config_file.write('num_cls: ' + str(args.num_cls) + '\n')
    config_file.write('batch_size: ' + str(args.batch_size) + '\n')
    config_file.write('optimizer: ' + 'Adam' + '\n')
    config_file.write('learning_rate: ' + str(args.init_lr) + ' : ' + str(args.lr_min) + '\n')
    config_file.write('data_augmentation: ' + str(args.data_augment) + '\n')
    config_file.write('max_angle: ' + str(args.max_angle) + '\n')
    if args.model == 'original_capsule':
        config_file.write('prim_caps_dim: ' + str(args.prim_caps_dim) + '\n')
        config_file.write('digit_caps_dim: ' + str(args.digit_caps_dim) + '\n')
    elif args.model == 'matrix_capsule':
        config_file.write('use_bias: ' + str(args.use_bias) + '\n')
        config_file.write('batch_normalization: ' + str(args.use_BN) + '\n')
        config_file.write('add_coords: ' + str(args.add_coords) + '\n')
        config_file.write('L2_reg: ' + str(args.L2_reg) + '\n')
        config_file.write('A: ' + str(args.A) + '\n')
        config_file.write('B: ' + str(args.B) + '\n')
        config_file.write('C: ' + str(args.C) + '\n')
        config_file.write('D: ' + str(args.D) + '\n')

    config_file.close()


def threshold_gui(image, centers, values, init_thresh=0, name=''):

    # create axes
    fig = plt.figure()
    plt.axis('off')
    plt.title(name)
    img_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

    # plot image and centers
    plt.axes(img_ax)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    dot_plt, = plt.plot(centers[:, 0], centers[:, 1], 'r.')
    a_slider = Slider(slider_ax, 'Thresh', 0, np.max(values), valinit=init_thresh)

    def update(val):
        print(val)
        val = a_slider.val
        pos_samples = np.where(values > val)[0]
        dot_plt.set_xdata(centers[pos_samples, 0])
        dot_plt.set_ydata(centers[pos_samples, 1])
        fig.canvas.draw()

    a_slider.on_changed(update)

    plt.show()
