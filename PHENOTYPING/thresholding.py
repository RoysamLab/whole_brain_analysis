import os
import numpy as np
import pandas as pd
from PIL import Image
import PIL.ImageDraw as ImageDraw
from skimage.io import imread_collection, imsave, imread
import matplotlib.pyplot as plt


def center_image(file_name, centers, image_size, r=2, color=None):
    '''
    Save RGB image with centers
    :param file_name: tifffile to be saved
    :param centers: np.array [centroid_x centroid_y]
    :param image_size: [width height]
    :param r : radius of center
    :param color: color of center 'red', 'blue', None: gray image
    :return:
    '''

    if color is None:
        image = Image.new('L', image_size)
        color = 'white'
    else:
        image = Image.new('RGB', image_size)
    center_draw = ImageDraw.Draw(image)

    for center in centers:
        center_draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), fill=color)

    try:
        image.save(file_name)
    except:
        imsave(file_name, np.array(image), plugin='tifffile', bigtiff=True)


def generate_centers_image(classification_table_fname, input_dir, output_dir, center_size=2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get image info
    files = [file for file in os.listdir(input_dir) if '.tif' in file]
    image = imread_collection(os.path.join(input_dir, files[0]), plugin='tifffile')
    im_size = image[0].shape[:2][::-1]

    # read bbxs file
    assert os.path.isfile(classification_table_fname), '{} not found!'.format(classification_table_fname)
    # if file exist -> load
    table = pd.read_csv(classification_table_fname, sep=',')
    centers = table[['centroid_x', 'centroid_y']].values

    # generate image for all the cells
    center_image(os.path.join(output_dir, 'all.tif'), centers, im_size, r=center_size)

    for bioM, cls in table.iteritems():
        if bioM not in ['ID', 'centroid_x', 'centroid_y', 'xmin', 'ymin', 'xmax', 'ymax']:
            centers = table.loc[table[bioM] == 1, ['centroid_x', 'centroid_y']].values
            center_image(os.path.join(output_dir, bioM + '.tif'), centers, im_size, r=center_size)


def update_classification_table_with_thresholds(bbxs_table_fname, feature_table_fname, channel_info_fname, output_dir,
                                                write_centers_image=False, **kwargs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read bbxs file
    assert os.path.isfile(bbxs_table_fname), '{} not found!'.format(bbxs_table_fname)
    # if file exist -> load
    #TODO: Read all files with ',' separator
    bbxs_table = pd.read_csv(bbxs_table_fname, sep='\t')
    bbxs_table.set_index('ID', inplace=True)

    # read feature table file
    assert os.path.isfile(feature_table_fname), '{} not found!'.format(feature_table_fname)
    # if file exist -> load
    feature_table = pd.read_csv(feature_table_fname, sep=',')
    feature_table.set_index('ID', inplace=True)

    # check if ID of bbxs_table and feature_table matched
    assert bbxs_table.loc[:, ['centroid_x', 'centroid_y']].equals(feature_table.loc[:, ['centroid_x', 'centroid_y']]), \
        'bounding box and feature tables are not equal...'

    # read channel info file
    assert os.path.isfile(channel_info_fname), '{} not found!'.format(channel_info_fname)
    # if file exist -> load
    channel_info_table = pd.read_csv(channel_info_fname, sep=',')

    # threshold each biomarker based on defined dictionary
    for _, channel in channel_info_table.iterrows():
        if not np.isnan(channel.Threshold):
            bbxs_table[channel.Biomarker] = np.where(feature_table[channel.Biomarker].values >= channel.Threshold, 1, 0)

    # save classification table
    bbxs_table.to_csv(os.path.join(output_dir, 'classification_table.csv'))

    if write_centers_image:
        input_dir = os.path.dirname(bbxs_table_fname)
        classification_table_fname = os.path.join(output_dir, 'classification_table.csv')
        generate_centers_image(classification_table_fname, input_dir, output_dir, **kwargs)


if __name__ == '__main__':

    bbxs_table_fname = r'E:\10_plex_stroke_rat\detection_results\bbxs_detection.txt'
    channel_info_fname = r'E:\10_plex_stroke_rat\channel_info.csv'
    feature_table_fname = r'E:\10_plex_stroke_rat\classification_results\assotiative_features.csv'
    output_dir = r'E:\10_plex_stroke_rat\classification_results'

    update_classification_table_with_thresholds(bbxs_table_fname, feature_table_fname, channel_info_fname, output_dir,
                                                write_centers_image=True)


