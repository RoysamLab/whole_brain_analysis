import os
import time
import h5py
import argparse
import numpy as np
import pandas as pd
import multiprocessing
import skimage.io as io
from functools import partial
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def zero_pad(image, dim):
    """
    pad zeros to the image in the first and second dimension
    :param image: image array [width*height*channel]
    :param dim: new dimension
    :return: padded image
    """
    pad_width = ((np.ceil((dim - image.shape[0]) / 2), np.floor((dim - image.shape[0]) / 2)),
                 (np.ceil((dim - image.shape[1]) / 2), np.floor((dim - image.shape[1]) / 2)),
                 (0, 0))
    return np.pad(image, np.array(pad_width, dtype=int), 'constant')


def to_square(image):
    """
    pad zeros to the image to make it square
    :param image: image array [width*height*channel]
    :param dim: new dimension
    :return: padded image
    """
    dim = max(image.shape[:2])
    if image.shape[0] >= image.shape[1]:
        pad_width = ((0, 0),
                     (np.ceil((dim - image.shape[1]) / 2), np.floor((dim - image.shape[1]) / 2)),
                     (0, 0))
    else:
        pad_width = ((np.ceil((dim - image.shape[0]) / 2), np.floor((dim - image.shape[0]) / 2)),
                     (0, 0),
                     (0, 0))
    return np.pad(image, np.array(pad_width, dtype=int), 'constant')


def main(input_dir, bbxs_file, channel_names, output_dir, inside_box=[8000, 4000, 34000, 24000], topN=7000, margin=5,
         crop_size=(50, 50), parallel=True):
    """
    Prepare training dataset file (data.h5) to train CapsNet for cell type classification
    :param input_dir: Path to the input dir containing biomarker images
    :param bbxs_file: Path to the bbxs_detection.txt file generated from cell nuclei detectio module
    :param channel_names: List of filnames for channels in the order: [dapi, histone, neun, s100, olig2, iba1, reca1]
    :param output_dir: Path to save the h5 file
    :param inside_box: Select cell inside this box to skip cells in the border with false phenotyping
    :param topN: Select top N brightest cells in each channel
    :param margin: Add extra margin to each dimension to capture information in soma
    :param crop_size: Size of the bounding box after reshape (input to the network)
    :param parallel: Process the file in multiprocessing or not
    :return:
    """
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default

    # prepare dict for biomarkers
    # for not-existing channel put ''
    biomarkers = {'DAPI': channel_names[0],
                  'Histones': channel_names[1],
                  'NeuN': channel_names[2],
                  'S100': channel_names[3],
                  'Olig2': channel_names[4],
                  'Iba1': channel_names[5],
                  'RECA1': channel_names[6]}

    # read bbxs file
    assert os.path.isfile(bbxs_file), '{} not found!'.format(bbxs_file)
    # if file exist -> load
    bbxs_table = pd.read_csv(bbxs_file, sep='\t')
    bbxs = bbxs_table[['xmin', 'ymin', 'xmax', 'ymax']].values

    # get bounding boxes in the center of brain
    bbxs = bbxs[(bbxs[:, 0] >= inside_box[0]) & (bbxs[:, 2] <= inside_box[2]) &
                (bbxs[:, 1] >= inside_box[1]) & (bbxs[:, 3] <= inside_box[3])]

    # shuffle the bounding boxes
    permutation = np.random.permutation(bbxs.shape[0])
    bbxs = bbxs[permutation, :]

    # get images
    image_size = io.imread_collection(os.path.join(input_dir, biomarkers['DAPI']), plugin='tifffile')[0].shape
    images = np.zeros((image_size[0], image_size[1], len(biomarkers)), dtype=np.uint16)

    # for each biomarker read the image and replace the black image if the channel is defined
    for i, bioM in enumerate(biomarkers.keys()):
        if biomarkers[bioM] != "":
            images[:, :, i] = io.imread(os.path.join(input_dir, biomarkers[bioM]))

    # from utils import bbxs_image
    # bbxs_image('all.tif', bbxs, images[:, :, 0].shape[::-1])

    # crop image (with extra margin) to get each sample (cell)
    def get_crop(image, bbx, margin=0):
        return image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :]

    ################### GENERATE LABELS ###############################
    # calculate the intensity of each channel
    intensities = np.array([np.mean(get_crop(images, bbx), axis=(0, 1)) for bbx in bbxs])
    intensities = intensities[:, 2:]  # we don't need DAPI and Histones for classification
    # find top N cells with highest intensity
    top_cells_each = [(-intensities[:, i]).argsort()[:topN] for i in range(intensities.shape[1])]
    top_cells = np.unique(np.array(top_cells_each).flatten())

    intensities = intensities[top_cells, :]
    labels = (intensities == intensities.max(axis=1)[:, None]).astype(int)

    # Check how many samples in each class
    # for i, bioM in zip(range(labels.shape[1]), list(biomarkers.keys())[2:]):
    #     print('{}: {}'.format(bioM, np.sum(labels[:, i], axis=0)))

    ################### GENERATE IMAGES ###############################
    # update bounding boxes to keep the desired cells
    bbxs = bbxs[top_cells, :]
    # get the crops
    cells = [get_crop(images, bbx, margin=margin) for bbx in bbxs]
    # del images

    # zero pad to the maximum dim
    max_dim = np.max([cell.shape[:2] for cell in cells])  # find maximum in each dimension
    if parallel:
        zero_pad_x = partial(zero_pad, dim=max_dim)
        with multiprocessing.Pool(processes=cpus) as pool:
            new_cells = pool.map(zero_pad_x, cells)
    else:
        new_cells = [zero_pad(cell, max_dim) for cell in cells]

    # resize image specific size
    if parallel:
        resize_x = partial(resize, output_shape=crop_size, mode='constant', preserve_range=True)
        with multiprocessing.Pool(processes=cpus) as pool:
            new_new_cells = pool.map(resize_x, new_cells)
    else:
        new_new_cells = [resize(cell, crop_size, mode='constant', preserve_range=True) for cell in new_cells]

    # visualize
    # id = 1400
    # fig = plt.figure(figsize=(10, 2))
    # for i in range(7):
    #     plt.subplot(1, 7, i + 1)
    #     plt.imshow(new_new_cells[id][:, :, i], cmap='gray', vmin=0, vmax=np.max(cells[id]))
    #     plt.title(list(biomarkers.keys())[i])
    # plt.tight_layout()
    # fig.suptitle('LABEL = {}'.format(list(biomarkers.keys())[np.argmax(labels[id])+2]))
    # plt.show()

    cells = np.array(new_new_cells)

    # visualize
    # from utils import bbxs_image
    # bbxs_image(biomarkers[2] + '.tif', bbxs[labels[:, 0] == 1, :], images[:, :, 3].shape[::-1], color='red')

    # split dataset to train, test and validation
    X_train, X_test, y_train, y_test = train_test_split(cells, labels, test_size=0.2)

    # save dataset in output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with h5py.File(os.path.join(output_dir, 'data.h5'), 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('Y_train', data=y_train)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('Y_test', data=y_test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_DIR', type=str, help='/path/to/input/dir')
    parser.add_argument('--OUTPUT_DIR', type=str, help='/path/to/output/dir')
    parser.add_argument('--BBXS_FILE', type=str, help='/path/to/bbxs/file')
    parser.add_argument('--DAPI', type=str, default='', help='<dapi.tif> | None')
    parser.add_argument('--HISTONES', type=str, default='', help='<histones.tif> | None')
    parser.add_argument('--NEUN', type=str, default='', help='<NeuN.tif> | None')
    parser.add_argument('--S100', type=str, default='', help='<S100.tif> | None')
    parser.add_argument('--OLIG2', type=str, default='', help='<Olig2.tif> | None')
    parser.add_argument('--IBA1', type=str, default='', help='<Iba1.tif> | None')
    parser.add_argument('--RECA1', type=str, default='', help='<RECA1.tif> | None')

    args = parser.parse_args()

    start = time.time()
    main(args.INPUT_DIR, args.BBXS_FILE, [args.DAPI, args.HISTONES, args.NEUN, args.S100, args.OLIG2, args.IBA1, args.RECA1],
         args.OUTPUT_DIR, inside_box=[8000, 4000, 34000, 24000], parallel=True, margin=5, crop_size=(50, 50), topN=5000)
    print('*' * 50)
    print('*' * 50)
    duration = time.time() - start
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    print('Data prepared for training in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))
    print('Data saved in {}'.format(args.OUTPUT_DIR))
