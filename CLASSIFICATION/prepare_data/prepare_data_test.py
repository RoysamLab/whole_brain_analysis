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


def get_crop(image, bbx, margin=0):
    """
    crop large image with extra margin
    :param image: large image
    :param bbx: [xmin, ymin, xmax, ymax]
    :param margin: margin from each side
    :return:
    """
    return image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin, :]


def main(input_dir, bbxs_file, channel_names, output_dir, margin=5, crop_size=(50, 50), parallel=True):
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
    if ".txt" in bbxs_file:
        bbxs_table = pd.read_csv(bbxs_file, sep='\t')
    else:   # .csv
        bbxs_table = pd.read_csv(bbxs_file)[["ID","centroid_x","centroid_y",'xmin', 'ymin', 'xmax', 'ymax']]
    bbxs = bbxs_table[['xmin', 'ymin', 'xmax', 'ymax']].values

    # get images
    image_size = io.imread_collection(os.path.join(input_dir, biomarkers['DAPI']), plugin='tifffile')[0].shape
    images = np.zeros((image_size[0], image_size[1], len(biomarkers)), dtype=np.uint16)

    # for each biomarker read the image and replace the black image if the channel is defined
    for i, bioM in enumerate(biomarkers.keys()):
        if biomarkers[bioM] != "":
            images[:, :, i] = io.imread(os.path.join(input_dir, biomarkers[bioM]))

    # Generate dataset
    X = [get_crop(images, bbx, margin=margin) for bbx in bbxs]
    del images

    # calculate mean intensity of each image -> we need it later for generating labels
    meanInt = np.array([np.mean(x, axis=(0, 1)) for x in X])
    meanInt = meanInt[:, 2:]  # we don't need DAPI and Histones for classification

    ## preprocess
    # zero pad to the maximum dim
    max_dim = np.max([cell.shape[:2] for cell in X]) # find maximum in each dimension
    if parallel:
        zero_pad_x = partial(zero_pad, dim=max_dim)
        with multiprocessing.Pool(processes=cpus) as pool:
            X = pool.map(zero_pad_x, X)
    else:
        X = [zero_pad(cell, max_dim) for cell in X]

    # resize image specific size
    if parallel:
        resize_x = partial(resize, output_shape=crop_size, mode='constant', preserve_range=True)
        with multiprocessing.Pool(processes=cpus) as pool:
            X = pool.map(resize_x, X)
    else:
        X = [resize(cell, crop_size, mode='constant', preserve_range=True) for cell in X]

    # Generate test set
    X_test = np.array(X)
    Y_test = np.zeros_like(meanInt, dtype=int)
    Y_test[np.arange(len(meanInt)), meanInt.argmax(1)] = 1

    # save dataset in output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # import pdb; pdb.set_trace()
    with h5py.File(os.path.join(output_dir, 'data.h5'), 'w') as f:
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('Y_test', data=Y_test)
        f.create_dataset('bbxs', data=bbxs_table)
        f.create_dataset('image_size', data=image_size[::-1])
        f.create_dataset('biomarkers', data=[x.encode('UTF8') for x in biomarkers])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_DIR', type=str, help='/path/to/input/dir')
    parser.add_argument('--OUTPUT_DIR', type=str, help='/path/to/output/dir')
    parser.add_argument('--BBXS_FILE', type=str, help='/path/to/bbxs_detection.txt')
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
         args.OUTPUT_DIR, margin=5, crop_size=(50, 50), parallel=True)
    print('*' * 50)
    print('*' * 50)
    duration = time.time() - start
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    print('Data prepared for inference in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))
    print('Data saved in {}'.format(args.OUTPUT_DIR))
