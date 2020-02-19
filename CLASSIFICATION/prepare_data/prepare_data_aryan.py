import os
import time
import h5py
import numpy as np
import pandas as pd
import multiprocessing
import skimage.io as io
from functools import partial
from skimage.transform import resize

input_dir = r'E:\50_plex\tif\pipeline2\final'
bbxs_file = r'E:\50_plex\tif\pipeline2\classification_results\regions\regions_table.csv'
parallel = True

width = 100
crop_size = 100


# for not-existing channel put ''
biomarkers = {'DAPI': 'S1_R2C1.tif',
              'Histones': 'S1_R2C2.tif',
              'NeuN': 'S1_R2C4.tif'
              # 'S100': 'S1_R3C5.tif',
              # 'Olig2': 'S1_R1C9.tif',
              # 'Iba1': 'S1_R1C5.tif',
              # 'RECA1': 'S1_R1C6.tif',
              # 'TH': 'S1_R1C2.tif',
              # 'BBB': 'S1_R1C3.tif',
              # 'GFP': 'S1_R1C4.tif',
              # 'Aquaporin-4': 'S1_R1C7.tif',
              # 'TomatoLectin': 'S1_R1C8.tif',
              # 'PDGFR': 'S1_R2C3.tif',
              # 'Synaptohysin': 'S1_R2C5.tif',
              # 'Parvalbumin': 'S1_R2C6.tif',
              # 'Choline Acetyltransferase': 'S1_R2C7.tif',
              # 'Glutaminase': 'S1_R2C9.tif',
              # 'SMA': 'S1_R2C10.tif',
              # 'GFAP': 'S1_R3C3.tif',
              # 'Doublecortin': 'S1_R3C4.tif',
              # 'Nestin': 'S1_R3C6.tif',
              # 'GLAST': 'S1_R3C9.tif',
              # 'Vimentin': 'S1_R3C10.tif',
              # 'Sox2': 'S1_R4C5.tif',
              # 'GAD67': 'S1_R4C6.tif',
              # 'Tbr1': 'S1_R4C9.tif',
              # 'Eomes': 'S1_R4C10.tif',
              # 'Calretinin': 'S1_R5C3.tif',
              # 'CNPase': 'S1_R5C4.tif',
              # 'Neurofilament-H': 'S1_R5C5.tif',
              # 'Myelin Basic Protein': 'S1_R5C6.tif',
              # 'Neurofilament-M': 'S1_R5C7.tif',
              # 'MAP2': 'S1_R5C9.tif',
              # 'Calbindin': 'S1_R5C10.tif'
              }


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


def get_crop(image, center, width=50):
    """
    crop large image with extra margin
    :param image: large image
    :param center: [centroid_x, centroid_y]
    :param margin: width of square
    :return:
    """
    return image[center[1] - width//2: center[1] + width//2, center[0] - width//2: center[0] + width//2]


def main():
    if parallel:
        try:
            cpus = multiprocessing.cpu_count()
        except NotImplementedError:
            cpus = 2  # arbitrary default
        pool = multiprocessing.Pool(processes=cpus)

    # read bbxs file
    assert os.path.isfile(bbxs_file), '{} not found!'.format(bbxs_file)
    # if file exist -> load
    bbxs_table = pd.read_csv(bbxs_file, sep=',')
    centers = bbxs_table[['centroid_x', 'centroid_y']].values

    # for each biomarker read the image and replace the black image if the channel is defined
    X = []
    for i, bioM in enumerate(biomarkers.keys()):
            image = io.imread(os.path.join(input_dir, biomarkers[bioM]), plugin='tifffile')
            # Generate dataset
            ch = [get_crop(image, center, width=width) for center in centers]

            # resize image specific size
            if crop_size != width:
                if parallel:
                    resize_x = partial(resize, output_shape=crop_size, mode='constant', preserve_range=True)
                    ch = pool.map(resize_x, X)
                else:
                    ch = [resize(cell, crop_size, mode='constant', preserve_range=True) for cell in X]

            # save to final list
            X.append(ch)

    ch = []
    image = []

    X_train = np.array(X)
    X_train = np.moveaxis(X_train, 0, -1)


    if parallel:
        pool.close()

    Y_train = np.array(bbxs_table['region'].values)

    np.savez_compressed('data_aryan.npz', x=X_train, y=Y_train)

    # with h5py.File('data_aryan.h5', 'w') as f:
    #     f.create_dataset('X_train', data=X_train)
    #     f.create_dataset('Y_train', data=[x.encode('UTF8') for x in Y_train])

if __name__ == '__main__':
    start = time.time()
    main()
    print('*' * 50)
    print('*' * 50)
    print('Pipeline finished successfully in {} seconds.'.format(time.time() - start))
