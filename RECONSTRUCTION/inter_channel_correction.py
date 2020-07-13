import os
import re
import sys
import time
import argparse
import warnings
import tifffile
import progressbar
import numpy as np
import pandas as pd
from scipy import optimize
from shutil import copyfile
from skimage.exposure import rescale_intensity, is_low_contrast
from scipy import linalg as LA
from sklearn import linear_model
from skimage import img_as_float, img_as_uint, img_as_ubyte


def imadjust(image, levels=None):
    if levels:
        if image.dtype == 'uint8':
            return rescale_intensity(image, in_range=(levels[0], levels[1]))
        elif image.dtype == 'uint16':
            return rescale_intensity(image, in_range=(levels[0]*257.0, levels[1]*257.0))  # * 65535/255
    else:
        return rescale_intensity(image)


def get_unmixing_channel_names_and_values(alphas, round_files):
    def append_nones(length, list_):
        """
        Appends Nones to list to get length of list equal to `length`.
        If list is too long raise AttributeError
        """
        diff_len = length - len(list_)
        if diff_len < 0:
            raise AttributeError('Length error list is too long.')
        return list_ + [None] * diff_len

    # get idx and value of channels
    script_rows_idx = [np.nonzero(row)[0] for row in alphas]
    script_rows_values = [row[np.nonzero(row)] for row in alphas]

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
                script_rows_values[row_idx] = [value for value, _ in sorted(zip(alphas[row_idx], round_files),
                                                                            reverse=True)[:3]]
        else:
            script_rows_str[row_idx] = [None] * 3

    return script_rows_str, script_rows_values


def calculate_unmixing_params_unsupervised(images):
    # convert to float
    images = img_as_float(images)
    num_channels = images.shape[2]

    # make a list to keep result parameters
    results = np.zeros((num_channels, num_channels))

    # for all channels
    for i in range(num_channels):
        endmembers = [np.ndarray.flatten(images[:, :, idx]) for idx in range(num_channels)]
        source = endmembers.pop(i)

        clf = linear_model.Lasso(alpha=.0001, copy_X=True, positive=True)
        clf.fit(np.array(endmembers).T, source)
        alphas = np.insert(clf.coef_, i, 0)

        results[i, :] = alphas

    return results


def calculate_unmixing_params_supervised(source, noises):
    # convert to float
    source = img_as_float(source)
    noises = img_as_float(noises)

    # define optimization function
    def f(params):
        alpha1, alpha2, alpha3 = params
        l1 = np.array(source - alpha1 * noises[:, :, 0]).flatten()
        l2 = np.array(source - alpha2 * noises[:, :, 1]).flatten()
        l3 = np.array(source - alpha3 * noises[:, :, 2]).flatten()
        return LA.norm(l1, 2) + LA.norm(l2, 2) + LA.norm(l3, 2)

    # run optimization
    initial_guess = [1, 1, 1]
    result = optimize.minimize(f, initial_guess, bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    alphas = result.x

    return alphas


def inter_channel_correct_supervised(input_dir, output_dir, script_file):
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read script
    df = pd.read_csv(script_file, index_col='filename')

    bar = progressbar.ProgressBar(max_value=df.shape[0])
    for idx, (src_name, src_info) in enumerate(df.iterrows()):
        bar.update(idx)
        # intra channel correction (unmixing)
        if src_info['inter channel correction'].lower() == 'yes':
            # get the coordinates of roi to calculate the unmixing parameters
            xmin, ymin, xmax, ymax = src_info['xmin'], src_info['ymin'], src_info['xmax'], src_info['ymax']

            # read source roi
            # TODO: fix memmap bug with offset is None
            # src_roi = tifffile.memmap(os.path.join(input_dir, src_name))
            src_roi = tifffile.imread(os.path.join(input_dir, src_name))[ymin:ymax, xmin:xmax]

            # TODO: extend the noise channels to variable (not 3)
            # read noise rois
            n_rois = np.zeros((src_roi.shape[0], src_roi.shape[1], 3), dtype=src_roi.dtype)
            for i in range(3):
                channel_name = src_info['channel {}'.format(i + 1)]
                if channel_name == channel_name:  # is not NaN
                    # TODO: fix memmap bug with offset is None
                    # n_rois[:, :, i] = tifffile.memmap(os.path.join(input_dir, channel_name))[ymin:ymax, xmin:xmax]
                    n_rois[:, :, i] = tifffile.imread(os.path.join(input_dir, channel_name))[ymin:ymax, xmin:xmax]

            # calculate unmixing parameters
            alphas = calculate_unmixing_params_supervised(src_roi, n_rois)
            n_rois = []  # free memory for noise rois

            # read source image and remove artifacts by zeroing pixels brighter than the brightest pixel in ROI
            source = tifffile.imread(os.path.join(input_dir, src_name))
            img_type = source.dtype

            # TODO: disable temporary -> come up with solid approach
            # clean artifacts in source image
            # source[source > np.max(src_roi)] = 0

            source = img_as_float(source)  # convert to float for subtraction
            src_roi = []

            for i in range(3):
                channel_name = src_info['channel {}'.format(i + 1)]
                if channel_name == channel_name:
                    noise = img_as_float(tifffile.imread(os.path.join(input_dir, channel_name)))
                    source -= (alphas[i] * noise)

            source[source < 0] = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if img_type == 'uint8':
                    source = img_as_ubyte(source)
                elif img_type == 'uint16':
                    source = img_as_uint(source)
                else:
                    sys.exit('image type not uint8 or uint16')

            # if level column in script and levels specified -> update level
            levels = None
            if 'level' in src_info:
                # if levels specified
                if src_info['level'] == src_info['level']:
                    levels = list(map(int, src_info['level'].split(',')))
            source = imadjust(source, levels=levels)

            # for low contrast images, adjust histogram to 99.99999% of the input range
            if is_low_contrast(source):
                _, p_up = np.percentile(source, (1, 99.99999))
                source = rescale_intensity(source, in_range=(0, p_up))

        # if no inter correction -> just level (if defined)
        elif src_info['inter channel correction'].lower() == 'no':
            # if no level column in script -> just copy and paste
            if 'level' not in src_info:
                copyfile(os.path.join(input_dir, src_name), os.path.join(output_dir, src_name))
                continue
            # if level column in script
            else:
                # if no level for channel
                if src_info['level'] != src_info['level']:
                    copyfile(os.path.join(input_dir, src_name), os.path.join(output_dir, src_name))
                    continue
                # if level for channel apply image adjust based on Photoshop level
                else:
                    source = tifffile.imread(os.path.join(input_dir, src_name))
                    levels = list(map(int, src_info['level'].split(',')))
                    source = imadjust(source, levels=levels)

        # save image
        save_name = os.path.join(output_dir, src_name)
        tifffile.imsave(save_name, source, bigtiff=True)


def inter_channel_correct_unsupervised(input_dir, output_dir, script_file):
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(script_file, index_col='filename')

    # copy the files with no correction from input to output dir
    for file in df.index[df['inter channel correction'] == 'No']:
        copyfile(os.path.join(input_dir, file), os.path.join(output_dir, file))

    # new df for channels with correction
    df = df[df['inter channel correction'] == 'Yes']
    files = df.index

    # find how many rounds and channels we have
    rounds = sorted(list(set([int(re.compile('R' + '(\d+)').findall(file)[0]) for file in files])))
    channels = sorted(list(set([int(re.compile('C' + '(\d+)').findall(file)[0]) for file in files])))

    # run unmixing for each round
    for round_idx in rounds:
        round_files = list(filter(lambda x: 'R' + str(round_idx) + 'C' in x, files))

        # get the default crop from first channel
        xmin, ymin, xmax, ymax = df.loc[df.index[0], ['xmin', 'ymin', 'xmax', 'ymax']]

        # read ROIs
        # TODO: fix memmap bug with offset is None
        # temp = tifffile.memmap(os.path.join(input_dir, round_files[1]))[ymin:ymax, xmin:xmax]
        temp = tifffile.imread(os.path.join(input_dir, round_files[1]))[ymin:ymax, xmin:xmax]
        rois = np.zeros((temp.shape[0], temp.shape[1], len(round_files)), dtype=temp.dtype)
        for i, filename in enumerate(round_files):
            # TODO: fix memmap bug with offset is None
            # rois[:, :, i] = tifffile.memmap(os.path.join(input_dir, filename))[ymin:ymax, xmin:xmax]
            rois[:, :, i] = tifffile.imread(os.path.join(input_dir, filename))[ymin:ymax, xmin:xmax]

        # calculate unmixing parameters using LASSO
        alphas = calculate_unmixing_params_unsupervised(rois)
        rois = []

        # write paramteres to the script
        channel_names, channel_values = get_unmixing_channel_names_and_values(alphas, round_files)
        for file, channels in zip(round_files, channel_names):
            df.loc[file, ['channel 1', 'channel 2', 'channel 3']] = channels
        df.to_csv(script_file)

        # unmix original images
        for filename, channels, values in zip(round_files, channel_names, channel_values):

            # read source image
            source = tifffile.imread(os.path.join(input_dir, filename))
            img_type = source.dtype
            source = img_as_float(source)  # convert to float for subtraction

            for i in range(len(values)):
                channel_name = channels[i]
                if channel_name == channel_name:
                    noise = img_as_float(tifffile.imread(os.path.join(input_dir, channel_name)))
                    source -= (values[i] * noise)

            source[source < 0] = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if img_type == 'uint8':
                    source = img_as_ubyte(source)
                elif img_type == 'uint16':
                    source = img_as_uint(source)
                else:
                    sys.exit('image type not uint8 or uint16')

            # rescale histogram of image
            source = rescale_intensity(source)

            # for low contrast images, adjust histogram to 99.99999% of the input range
            if is_low_contrast(source):
                _, p_up = np.percentile(source, (1, 99.99999))
                source = rescale_intensity(source, in_range=(0, p_up))


            # save image
            save_name = os.path.join(output_dir, filename)
            tifffile.imsave(save_name, source, bigtiff=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=r'unsupervised', help='mode (supervised | unsupervised)')
    parser.add_argument('--input_dir', type=str, help='path to the directory of images')
    parser.add_argument('--output_dir', type=str, help='path to the directory to save unmixed images')
    parser.add_argument('--script_file', type=str, default=r'../scripts/20_plex.csv', help='script file name')
    parser.add_argument('--brightfield', type=int, default=11, help='Channel number for brightfield')
    parser.add_argument('--default_crop', type=int, nargs='+', default=[34000, 8000, 40000, 12000], help='[xmin ymin xmax ymax]')
    args = parser.parse_args()

    start = time.time()
    if args.mode == 'unsupervised':
        from prepare_script import create_script
        create_script(os.path.join(args.output_dir, 'script.csv'), args.input_dir, args.default_crop, brightfield=args.brightfield)
        inter_channel_correct_unsupervised(args.input_dir, args.output_dir, os.path.join(args.output_dir, 'script.csv'))
    elif args.mode == 'supervised':
        inter_channel_correct_supervised(args.input_dir, args.output_dir, args.script_file)

    duration = time.time() - start
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    print('Intra channel correction finished successfully in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))

    # TODO: save to overwright on input script
