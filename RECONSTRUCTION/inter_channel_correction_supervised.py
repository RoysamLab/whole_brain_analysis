import os
import time
import argparse
import warnings
import pandas as pd
import numpy as np
from scipy import linalg as LA
from scipy import optimize
from skimage import img_as_float, img_as_uint
from skimage import exposure
import tifffile
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default=r'E:\50_plex\tif\pipeline2\IL_corrected', help='path to the directory of images')
parser.add_argument('--output_dir', type=str, default=r'E:\50_plex\tif\pipeline2\unmixed', help='path to the directory to save unmixed images')
parser.add_argument('--script_file', type=str, default=r'E:\50_plex\tif\pipeline2\unmixed\scripts\supervised.csv', help='script file name')
parser.add_argument('--default_box', type=str, default='10000_7000_20000_12000', help='xmin_ymin_xmax_ymax')
parser.add_argument('--visualize', type=bool, default=False, help='plot the unmixing report | True | False')
args = parser.parse_args()


input_dir = args.input_dir


def visualize_results(s, n1, n2, n3, c, name):

    fig = plt.figure()

    ax1 = fig.add_subplot(221)
    ax1.imshow(s, cmap='gray')
    ax1.axis('off')
    ax1.set_title('source')

    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    ax2.imshow(c, cmap='gray')
    ax2.axis('off')
    ax2.set_title('unmixed')

    ax3 = fig.add_subplot(234, sharex=ax1, sharey=ax1)
    ax3.imshow(n1, cmap='gray')
    ax3.axis('off')
    ax3.set_title('noise 1')

    ax4 = fig.add_subplot(235, sharex=ax1, sharey=ax1)
    ax4.imshow(n2, cmap='gray')
    ax4.axis('off')
    ax4.set_title('noise 2')

    ax5 = fig.add_subplot(236, sharex=ax1, sharey=ax1)
    ax5.imshow(n3, cmap='gray')
    ax5.axis('off')
    ax5.set_title('noise 3')



    plt.suptitle(name)
    plt.savefig(name + '.png', dpi=1000)
    plt.close(fig)


def unmix_channel(src_name, n1_name, n2_name, n3_name, box_info, visualize=False):

    # load images
    source = img_as_float(tifffile.imread(os.path.join(input_dir, src_name)))

    if str(n1_name) == 'nan':
        noise_1 = np.empty_like(source)
    else:
        noise_1 = img_as_float(tifffile.imread(os.path.join(input_dir, n1_name)))
    if str(n2_name) == 'nan':
        noise_2 = np.empty_like(source)
    else:
        noise_2 = img_as_float(tifffile.imread(os.path.join(input_dir, n2_name)))

    if str(n3_name) == 'nan':
        noise_3 = np.empty_like(source)
    else:
        noise_3 = img_as_float(tifffile.imread(os.path.join(input_dir, n3_name)))

    xmin = box_info[0]
    ymin = box_info[1]
    xmax = box_info[2]
    ymax = box_info[3]

    # create small crop for optimization
    s = source[ymin:ymax, xmin:xmax]
    n1 = noise_1[ymin:ymax, xmin:xmax]
    n2 = noise_2[ymin:ymax, xmin:xmax]
    n3 = noise_3[ymin:ymax, xmin:xmax]

    # define optimization function
    def f(params):
        alpha1, alpha2, alpha3 = params
        l1 = np.array(s - alpha1 * n1).flatten()
        l2 = np.array(s - alpha2 * n2).flatten()
        l3 = np.array(s - alpha3 * n3).flatten()
        return LA.norm(l1, 2) + LA.norm(l2, 2) + LA.norm(l3, 2)

    # optimize paramteres
    initial_guess = [1, 1, 1]
    result = optimize.minimize(f, initial_guess, bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    alpha1, alpha2, alpha3 = result.x
    print('\n'.join('alpha_{} = {:.2f}'.format(i, alpha) for i, alpha in enumerate(result.x)))

    # apply to crop
    c = s - alpha1 * n1 - alpha2 * n2 - alpha3 * n3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = img_as_uint(c)

    # visualize crop
    if visualize:
        visualize_results(s=s, n1=n1, n2=n2, n3=n3, c=c, name=os.path.splitext(src_name)[0])

    # clean artifacts in image
    max_s = np.max(s)
    source[source > max_s] = 0

    # create unmixed image
    corrected_img = source - alpha1 * noise_1 - alpha2 * noise_2 - alpha3 * noise_3
    corrected_img[corrected_img < 0] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corrected_img = img_as_uint(corrected_img)

    return corrected_img, [alpha1, alpha2, alpha3]


def rescale_histogram(image):
    return exposure.rescale_intensity(image, in_range='image', out_range='dtype')


def main():
    # create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    default_box = list(map(int, args.default_box.split('_')))

    # read script
    df = pd.read_csv(args.script_file)

    # add/update new alphas for each channel
    df["alpha1"] = np.nan if 'alpha1' not in df.columns else None
    df["alpha2"] = np.nan if 'alpha2' not in df.columns else None
    df["alpha3"] = np.nan if 'alpha3' not in df.columns else None

    for index, row in df.iterrows():
        src_name = row["filename"]

        print('*' * 50)
        print('unmixing image {}'.format(src_name))

        n1_name = str(row["channel_1"])
        n2_name = str(row["channel_2"])
        n3_name = str(row["channel_3"])

        # if no noise channel is given just rescale the histogram
        if all([x == 'nan' for x in[n1_name, n2_name, n3_name]]):
            # for images without unmixing, just rescale the histogram
            unmixed_image = tifffile.imread(os.path.join(input_dir, src_name))
        else:
            # read information from box
            box_info = np.empty(4, dtype=int)
            box_info[0] = int(row["xmin"]) if pd.notna(row['xmin']) else default_box[0]     #xmin
            box_info[1] = int(row["ymin"]) if pd.notna(row['ymin']) else default_box[1]     #ymin
            box_info[2] = int(row["xmax"]) if pd.notna(row['xmax']) else default_box[2]     #xmax
            box_info[3] = int(row["ymax"]) if pd.notna(row['ymax']) else default_box[3]     #ymax

            # unmix
            unmixed_image, alpha = unmix_channel(src_name, n1_name, n2_name, n3_name, box_info, visualize=args.visualize)

        # update alphas with derived values
        df.loc[index, 'alpha1'] = alpha[0] if str(n1_name) != 'nan' else 0
        df.loc[index, 'alpha2'] = alpha[1] if str(n2_name) != 'nan' else 0
        df.loc[index, 'alpha3'] = alpha[2] if str(n3_name) != 'nan' else 0

        # rescale histogram of image
        adjusted_img = rescale_histogram(unmixed_image)

        # save image
        save_name = os.path.join(args.output_dir, src_name)
        tifffile.imsave(save_name, adjusted_img, bigtiff=True)

    df.to_csv(args.script_file, index=False)


if __name__ == '__main__':

    start = time.time()
    main()
    print('*' * 50)
    print('*' * 50)
    print('Unmixing pipeline finished successfully in {} seconds.'.format(time.time() - start))

    #TODO: save to overwright on input script
