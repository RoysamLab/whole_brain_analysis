import os
import cv2
import time
import argparse
import progressbar
import numpy as np
import pandas as pd
from shutil import copyfile
from skimage.io import imread, imsave
from skimage import exposure



def rescale_histogram(image):
    return exposure.rescale_intensity(image, in_range='image', out_range='dtype')


def intra_channel_correct(input_dir, output_dir, disk_size, script_file=None, brightfield=None):
    """
    :param input_dir: path to the directory to read input images
    :param output_dir: path to the directory to save output images
    :param disk_size: diameter of smallest and largest objects
    :param script_file: path to the script file to read channels
    :param brightfield: brightfiled channel number if script_file is not defined
    """

    # create dir to save images if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get files for correction from script or input directory
    if script_file:
        df = pd.read_csv(script_file, index_col='filename')
        files = df.index[df['intra channel correction'] == 'Yes']

        # copy the files with no correction from input to output dir
        for file in df.index[df['intra channel correction'] == 'No']:
            copyfile(os.path.join(input_dir, file), os.path.join(output_dir, file))
    else:
        # list all images in input dir
        files = os.listdir(input_dir)

        # remove brightfield from list and copy to output dir if defined
        if brightfield:
            # TODO: add other extensions
            bf_channels = list(filter(lambda x: 'C' + str(brightfield) + '.tif' in x, files))
            [copyfile(os.path.join(input_dir, file), os.path.join(output_dir, file)) for file in bf_channels]

            # remove channels from list of files
            files = [x for x in files if x not in bf_channels]

    bar = progressbar.ProgressBar(max_value=len(files))
    for idx, file in enumerate(files):
        # read the image
        im = imread(os.path.join(input_dir, file), plugin='tifffile')

        # create a background array similar to original image
        background = np.copy(im)
        for sz in disk_size:
            # apply morphological opening with the defined structuring element
            selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
            background = cv2.morphologyEx(background, cv2.MORPH_OPEN, selem)

        # subtract the background from image
        im = cv2.subtract(im, background)

        # normalize image 
        im_normalized = rescale_histogram(im)

        # save processed image to the disk
        imsave(os.path.join(output_dir, file), im_normalized, check_contrast=False, plugin='tifffile', bigtiff=True)
        bar.update(idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='path to the directory of input images')
    parser.add_argument('--output_dir', type=str, required=True, help='path to the directory to save corrected images')
    parser.add_argument('--disk_size', type=int, nargs='+', default=[40, 80], help='Diameters of smallest and largets objects')
    parser.add_argument('--script_file', type=str, help='script file name')
    parser.add_argument('--brightfield', type=int, help='Channel number for brightfield')

    args = parser.parse_args()

    start = time.time()
    intra_channel_correct(args.input_dir, args.output_dir, args.disk_size, script_file=args.script_file, brightfield=args.brightfield)
    duration = time.time() - start
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    print('Intra channel correction finished successfully in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))
