import os
import time
import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from lib.image_uitls import center_image

feature_table = r'E:\50_plex\tif\pipeline2\S1___FeatureTable.csv'
image_sample = r'E:\50_plex\tif\pipeline2\unmixed\S1_R1C1.tif'
save_dir = r'E:\50_plex\tif\pipeline2\detection_results\phenotyping'
biomarkers = {'NeuN': 1535.85,
              'Iba1': 1301.10,
              'Olig2': 1297.38,
              'RECA1': 1396.77,
              'S100': 2094.77,
              'Cleaved Caspase-3': 5633.84,
              'Tyrosine Hydroxylase': 2020.86,
              'Blood Brain Barrier': 696.96,
              'GFP(R1C4)': 2063.84,
              'PDGFR beta': 3141.93,
              'Parvalbumin': 3837.34,
              'Choline Acetyltransferase': 7199.8,
              'GFAP': 957.28,
              'Smooth Muscle Actin': 670.22,
              'Glutaminase': 5635.48,
              'Doublecortin': 3445.17,
              'Nestin': 2518.63,
              'Aquaporin-4': 2166.31,
              'Sox2': 4219.31,
              'PCNA': 6099.52,
              'Vimentin': 915.39,
              'GAD67': 1422.59,
              'Tbr1': 1866.82,
              'Eomes': 2018.98,
              'Calretinin': 4646.71,
              'CNPase': 3726.25
              }


def generate_centers_image():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get image info
    image = imread(image_sample)
    im_size = image.shape[::-1]

    # read bbxs file
    assert os.path.isfile(feature_table), '{} not found!'.format(feature_table)
    # if file exist -> load
    table = pd.read_csv(feature_table, sep=',')
    centers = table[['centroid_x', 'centroid_y']].values
    # ICE file requires different format, restore to original format
    centers[:, 1] = im_size[1] - centers[:, 1]

    for name, thresh in biomarkers.items():
        bioM = table[table[name] >= thresh]
        centers = bioM[['centroid_x', 'centroid_y']].values
        # ICE file requires different format, restore to original format
        centers[:, 1] = im_size[1] - centers[:, 1]
        center_image(os.path.join(save_dir, name+'.tif'), centers, im_size, color='red')


def generate_classification_table():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get image info
    image = imread(image_sample)
    im_size = image.shape[::-1]

    # read bbxs file
    assert os.path.isfile(feature_table), '{} not found!'.format(feature_table)
    # if file exist -> load
    table = pd.read_csv(feature_table, sep=',')
    centers = table[['centroid_x', 'centroid_y']].values
    # ICE file requires different format, restore to original format
    centers[:, 1] = im_size[1] - centers[:, 1]

    # threshold each biomarker based on defined dictionary
    for name, thresh in biomarkers.items():
        table[name] = np.where(table[name] >= thresh, 1, 0)

    # ICE file requires different format, restore to original format
    table['centroid_y'] = im_size[1] - table['centroid_y']

    # get list of defined biomakers to remove the redundant ones
    table = table[['centroid_x', 'centroid_y'] + [*biomarkers]]

    # set index column header
    table.index.name = 'ID'

    # save classification table
    table.to_csv(os.path.join(save_dir, 'classification_table.csv'))



if __name__ == '__main__':

    start = time.time()
    # generate_centers_image()
    generate_classification_table()
    print('*' * 50)
    print('*' * 50)
    print('Pipeline finished successfully in {} seconds.'.format(time.time() - start))
