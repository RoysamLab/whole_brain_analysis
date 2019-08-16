import os
import re
import time
import argparse
import warnings
import pandas as pd
import numpy as np
from skimage import img_as_float, img_as_uint
from skimage import exposure
import tifffile
import matplotlib.pyplot as plt
from sklearn import linear_model

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default=r'E:\jahandar\DashData\G3_BR#15_HC_12L\IL_corrected', help='path to the directory of input images')
parser.add_argument('--output_dir', type=str, default=r'E:\jahandar\DashData\G3_BR#15_HC_12L\unmixed', help='path to the directory to save unmixed images')
parser.add_argument('--brightfield', type=int, help='Channel number for brightfield')
parser.add_argument('--default_box', type=str, default='2000_4000_15000_13000', help='xmin_ymin_xmax_ymax')
parser.add_argument('--round_pattern', type=str, default='R', help='pattern for round idx')
parser.add_argument('--channel_pattern', type=str, default='C', help='pattern for channel idx')

args = parser.parse_args()

input_dir = args.input_dir


def rescale_histogram(image):
    return exposure.rescale_intensity(image, in_range='image', out_range='dtype')


def get_unmixing_params(images):
    # make a list to keep result parameters
    results = np.zeros((len(images), len(images)))

    # for all channels
    for i in range(1, len(images)):
        endmembers = [np.ndarray.flatten(img) for img in images]
        source = endmembers.pop(i)

        clf = linear_model.Lasso(alpha=.0001, copy_X=True, positive=True)
        clf.fit(np.array(endmembers).T, source)
        alphas = np.insert(clf.coef_, i, 0)

        results[i, :] = alphas

    return results


def write_params_to_csv(script, filename, alphas, round_files):

    def append_nones(length, list_):
        """
        Appends Nones to list to get length of list equal to `length`.
        If list is too long raise AttributeError
        """
        diff_len = length - len(list_)
        if diff_len < 0:
            raise AttributeError('Length error list is too long.')
        return list_ + [None] * diff_len

    # get idx of channels for each file in script
    script_rows_idx = [np.nonzero(row)[0] for row in alphas]
    # convert idx of channel to filename
    script_rows_str = []
    for row_idx, row in enumerate(script_rows_idx):
        script_rows_str.append([])
        if len(row) != 0:
            new_row = []
            for idx in row:
                new_row.append(round_files[idx])
            script_rows_str[row_idx].extend(new_row)
            if len(script_rows_str[row_idx]) < 3:
                script_rows_str[row_idx] = append_nones(3, script_rows_str[row_idx])
            if len(script_rows_str[row_idx]) > 3:
                script_rows_str[row_idx] = [fname for _, fname in sorted(zip(alphas[row_idx], round_files),
                                                                         reverse=True)[:3]]
        else:
            script_rows_str[row_idx] = [None] * 3

    # update script with unmixed channels
    script.loc[round_files, ['channel_1', 'channel_2', 'channel_3']] = script_rows_str

    # save script to disk
    script.to_csv(filename)

def unmix_original_images(rois, images, alphas, names):
    for roi, image, alpha, name in zip(rois, images, alphas, names):

        # clean artifacts in image
        max_s = np.max(roi)
        image[image > max_s] = 0

        # unmix whole brain image
        if np.sum(alpha) > 0.0:
            corrected_img = image - np.sum([a * img for a, img in zip(alpha, images) if a != 0.0], axis=0)
            corrected_img[corrected_img < 0] = 0
        else:
            corrected_img = image

        # extend shrank histogram
        corrected_img = rescale_histogram(corrected_img)

        # convert to uint16
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corrected_img = img_as_uint(corrected_img)

        # save image
        save_name = os.path.join(args.output_dir, name)
        tifffile.imsave(save_name, corrected_img, bigtiff=True)


def main():

    # read script from intra channel correction module
    script = pd.read_csv(os.path.join(input_dir, 'script.csv'), index_col='filename')
    script = script.reindex(columns=[*script.columns.tolist(), *['channel_1', 'channel_2', 'channel_3']])
    files = script.index.values

    # find how many rounds and channels we have
    rounds = sorted(list(set([int(re.compile(args.round_pattern + '(\d+)').findall(file)[0]) for file in files])))
    channels = sorted(list(set([int(re.compile(args.channel_pattern + '(\d+)').findall(file)[0]) for file in files])))

    default_box = list(map(int, args.default_box.split('_')))
    xmin, ymin, xmax, ymax = default_box

    images = []
    rois = []
    for round_idx in rounds:
        print('*' * 50)
        print('Unxminging round {} ...'.format(round_idx))
        round_files = list(filter(lambda x: args.round_pattern + str(round_idx) in x, files))

        if args.brightfield:
            # remove the brightfield channel
            brightfiled_name = args.channel_pattern + str(args.brightfield) + '.tif'
            round_files = [file for file in round_files if brightfiled_name not in file]

        # read images
        print('Reading images.')
        for filename in round_files:
            #TODO: switch to skimage.io.imread_collection
            # read image and append to list
            image = img_as_float(tifffile.imread(os.path.join(input_dir, filename)))
            images.append(image)
            rois.append(np.copy(image[ymin:ymax, xmin:xmax]))

        # find unmixing params from ROIs
        print('Calculating unmixing parameters from ROIs.')
        alphas = get_unmixing_params(rois)

        # create folder and save unmixing parameters into csv file
        print('writing unmixing parameters in {}'.format(os.path.join(args.output_dir, 'unmixing_script_unsupervised.csv')))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # save parameters in the parent directory
        parentdir = os.path.dirname(args.output_dir)
        write_params_to_csv(script, os.path.join(parentdir, 'script.csv'),
                            alphas, round_files)
        # save unmixed images
        print('Unmixing images and writing to disk')
        unmix_original_images(rois, images, alphas, round_files)

        images = []
        rois = []


if __name__ == '__main__':

    start = time.time()
    main()
    print('*' * 50)
    print('*' * 50)
    print('Unmixing pipeline finished successfully in {} seconds.'.format(time.time() - start))


    # TODO: script is fixed with size of 3 endmembers -> np.count_nonzero(alphas)
