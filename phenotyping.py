import os
import time
import numpy as np
import pandas as pd
import skimage.io as io
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.io import imread_collection, imsave, imread
from prepare_data.utils import center_image
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)


CELL_TYPES = ['Neuron', 'Astrocyte', 'Oligodendrocyte', 'Microglia', 'Blood vessel']

channel_info_fname = r'E:\50_plex\scripts\channel_info.csv'
boolean_table = r'E:\50_plex\scripts\Boolean_table_sheet_2.xlsx'
bbxs_table = r'E:\50_plex\tif\pipeline2\detection_results\bbxs_detection.txt'
feature_table = r'E:\50_plex\tif\pipeline2\detection_results\phenotyping\original\associated_features.csv'
classification_table_base = r'E:\jahandar\CapsNet\Results\data_dir\run02\classification_table.csv'
classification_table_full = r'E:\jahandar\CapsNet\Results\data_dir\run02\phenotyping\classification_table.csv'
image_sample = r'E:\50_plex\tif\pipeline2\final\S1_R1C1.tif'
image_dir = r'E:\50_plex\tif\pipeline2\final'
save_dir = r'E:\jahandar\CapsNet\Results\data_dir\run02\phenotyping'
center_size = 2
biomarker_dict = {'Cleaved Caspase-3': 3711.65,
                  'Tyrosine Hydroxylase': 4542.26,
                  'Blood Brain Barrier': 978.39,
                  'GFP': 2122.70,
                  'PDGFR beta': 3141.93,
                  'Parvalbumin': 3294.34,
                  'Choline Acetyltransferase': 4980.4,
                  'GFAP': 1754.4,
                  'Smooth Muscle Actin': 2038.9,
                  'Glutaminase': 5023.6,
                  'Doublecortin': 4459.0,
                  'Sox2': 3708.7,
                  'PCNA': 2972.9,
                  'Vimentin': 3971.6,
                  'GAD67': 2506.06,
                  'Tbr1': 1793.38,
                  'Eomes': 2018.98,
                  'Calretinin': 4646.71,
                  'Nestin': 1931.85,
                  'Aquaporin-4': 2524.47,
                  'Calbindin': 8056.71
                  }


def get_crop(image, bbx, margin=0):
    """
    crop large image with extra margin
    :param image: large image
    :param bbx: [xmin, ymin, xmax, ymax]
    :param margin: margin from each side
    :return:
    """
    return image[bbx[1] - margin:bbx[3] + margin, bbx[0] - margin:bbx[2] + margin]


def plot_loghist(x, bins, name):
    hist, bins, _ = plt.hist(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.hist(x, bins=logbins)
    plt.xscale('log')
    plt.title(name)
    plt.show()


def generate_centers_image():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get image info
    image = imread_collection(image_sample, plugin='tifffile')
    im_size = image[0].shape[::-1]

    # read bbxs file
    assert os.path.isfile(feature_table), '{} not found!'.format(feature_table)
    # if file exist -> load
    table = pd.read_csv(feature_table, sep=',')
    centers = table[['centroid_x', 'centroid_y']].values
    # ICE file requires different format, restore to original format
    centers[:, 1] = im_size[1] - centers[:, 1]

    # generate image for all the cells
    center_image(os.path.join(save_dir, 'all.tif'), centers, im_size, r=center_size)

    for name, thresh in biomarker_dict.items():
        bioM = table[table[name] >= thresh]
        centers = bioM[['centroid_x', 'centroid_y']].values
        # ICE file requires different format, restore to original format
        centers[:, 1] = im_size[1] - centers[:, 1]
        center_image(os.path.join(save_dir, name+'.tif'), centers, im_size, r=center_size)


def generate_classification_table():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get image info
    # image = imread(image_sample)
    # im_size = image.shape[::-1]

    # read bbxs file
    assert os.path.isfile(feature_table), '{} not found!'.format(feature_table)
    # if file exist -> load
    table = pd.read_csv(feature_table, sep=',')
    # centers = table[['centroid_x', 'centroid_y']].values
    # ICE file requires different format, restore to original format
    # centers[:, 1] = im_size[1] - centers[:, 1]

    # threshold each biomarker based on defined dictionary
    for name, thresh in biomarker_dict.items():
        table[name] = np.where(table[name] >= thresh, 1, 0)

    # ICE file requires different format, restore to original format
    # table['centroid_y'] = im_size[1] - table['centroid_y']

    # get list of defined biomakers to remove the redundant ones
    table = table[['centroid_x', 'centroid_y'] + [*biomarker_dict]]

    # set index column header
    table.index.name = 'ID'

    # save classification table
    table.to_csv(os.path.join(save_dir, 'classification_table.csv'))


def generate_boolean_center_image():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get image info
    image = imread_collection(image_sample, plugin='tifffile')
    im_size = image[0].shape[::-1]

    # read bbxs file
    assert os.path.isfile(classification_table_base), '{} not found!'.format(classification_table_base)
    # if file exist -> load
    class_table = pd.read_csv(classification_table_base, sep=',')
    class_table.set_index('ID', inplace=True)

    # # ICE file requires different format, restore to original format
    # centers = class_table[['centroid_x', 'centroid_y']].values
    # centers[:, 1] = im_size[1] - centers[:, 1]

    # read boolean file
    assert os.path.isfile(boolean_table), '{} not found!'.format(boolean_table)
    # if file exist -> load
    bool_table = pd.read_csv(boolean_table, sep=',')
    bool_table = bool_table.replace('+', 1.0)
    bool_table = bool_table.replace('-', 0.0)
    bool_table = bool_table.set_index('Type')

    # create table to write cell types
    cell_type_table = class_table.copy()[['centroid_x', 'centroid_y']]
    cell_type_table['cell_type'] = ''

    # for each cell type:
    for index, cell_type in bool_table.iterrows():

        # remove +/- from the list because we don't care
        cell_type = cell_type[cell_type != '+-']

        # find rows in classification table that match the cell_type row
        pos_cells = np.where((class_table[cell_type.keys()] == cell_type.values).all(1))[0]

        # update the cell type table with the cell type class
        cell_type_table.loc[pos_cells, 'cell_type'] = cell_type.name

        # find centers corresponding to rows
        centers = class_table.loc[pos_cells, ['centroid_x', 'centroid_y']].values

        # save the image
        center_image(os.path.join(save_dir, cell_type.name + '.tif'), centers, im_size, r=center_size)

    cell_type_table.to_csv(os.path.join(save_dir, 'cell_type_table.csv'))


def generate_boolean_center_image_2():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get image info
    image = imread_collection(image_sample)
    im_size = image.shape[::-1]

    # read bbxs file
    assert os.path.isfile(classification_table_base), '{} not found!'.format(classification_table_base)
    # if file exist -> load
    class_table = pd.read_csv(classification_table_base, sep=',')
    class_table = class_table.set_index('ID')

    # # ICE file requires different format, restore to original format
    # centers = class_table[['centroid_x', 'centroid_y']].values
    # centers[:, 1] = im_size[1] - centers[:, 1]

    # read boolean file
    assert os.path.isfile(boolean_table), '{} not found!'.format(boolean_table)
    # if file exist -> load
    bool_table = pd.read_csv(boolean_table, sep=',')


    # create table to write cell types
    cell_type_table = class_table.copy()[['centroid_x', 'centroid_y']]
    cell_type_table['type'] = ''
    cell_type_table['function'] = ''

    major_cells = {'Neuron': 'NeuN',
                   'Microglia': 'Iba1',
                   'Astrocyte': 'S100',
                   'Oligodendrocyte': 'Olig2',
                   'Endothelial cell': 'RECA1'}

    # for each cell type:
    for index, cell_type in bool_table.iterrows():
        print(cell_type['Cell Type'])
        # find that the cell type belongs to which major cell type
        cell_list = [cell if cell in cell_type['Cell Type'] else None for cell in major_cells.keys()]
        this_cell = [i for i in cell_list if i is not None][0]  # remove None values from list

        if cell_type['Cell Type'] in major_cells.keys():
            # find cells in classification table that is positive for biomarker
            pos_cells = np.where((class_table[cell_type['Biomarker']] == 1)
                                 & (cell_type_table['function'] == ''))[0]
            cell_type_table.loc[pos_cells, 'type'] = cell_type['Cell Type']
        else:
            pos_cells = np.where((class_table[major_cells[this_cell]] == 1)
                                 & (class_table[cell_type['Biomarker']] == 1))[0]
            cell_type_table.loc[pos_cells, 'function'] = cell_type['Cell Type']


        # find centers corresponding to cells
        # centers = class_table.loc[pos_cells, ['centroid_x', 'centroid_y']].values

        # save the image
        # center_image(os.path.join(save_dir, cell_type.name + '.tif'), centers, im_size, r=center_size)

    cell_type_table.to_csv(os.path.join(save_dir, 'temp.csv'))


def generate_boolean_image():
    """
    Generate Boolean image by combining positive channels from Boolean table
    """
    input_dir = os.path.split(image_sample)[0]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # read channel info table
    channel_info_table = pd.read_csv(channel_info_fname, sep=',')

    # read bbxs file
    assert os.path.isfile(classification_table_base), '{} not found!'.format(classification_table_base)


    # read boolean file
    assert os.path.isfile(boolean_table), '{} not found!'.format(boolean_table)
    # if file exist -> load
    bool_table = pd.read_csv(boolean_table, sep=',')
    bool_table = bool_table.replace('+', 1.0)
    bool_table = bool_table.replace('-', 0.0)
    bool_table = bool_table.set_index('Type')

    # read feature table
    feature_table_table = pd.read_csv(feature_table, sep=',')

    # for each cell type:
    for index, cell_type in bool_table.iterrows():
        # find positive biomarkers
        bioM_list = list(cell_type[cell_type.values == 1.0].keys())
        # find channels corresponding positive biomarkers
        channel_list = list(channel_info_table.loc[channel_info_table['Biomarker'].isin(bioM_list)]['Channel'])
        channel_list = [os.path.join(input_dir, file) for file in channel_list]
        # read channels corresponding positive biomarkers
        im_collection = io.ImageCollection(channel_list)
        images = io.collection.concatenate_images(im_collection).astype(float)

        # if more than one image sum channels, if not use the single channel
        if images.shape[0] > 1:
            image = (np.sum(images, axis=0)/images.shape[0]).astype('uint16')
            image = rescale_intensity(image)
        else:
            image = images[0].astype('uint16')

        # save image
        imsave(os.path.join(save_dir, cell_type.name + '.tif'), image, plugin='tifffile', bigtiff=True)


def update_classification_table():
    table = pd.read_csv(classification_table_base)
    table.set_index('ID', inplace=True)

    # read channel info table
    channel_info = pd.read_csv(channel_info_fname, sep=',')

    for bioM, thresh in biomarker_dict.items():
        channel_name = channel_info.loc[channel_info['Biomarker'] == bioM, 'Channel'].values[0]
        image = imread(os.path.join(image_dir, channel_name))
        bbxs = table[['xmin', 'ymin', 'xmax', 'ymax']].values

        X = [get_crop(image, bbx) for bbx in bbxs]
        meanInt = np.array([np.mean(x) for x in X])

        table[bioM] = np.where(meanInt > thresh, 1, 0)

        # # temp
        # from skimage.transform import downscale_local_mean
        # import matplotlib.pyplot as plt
        # from matplotlib.widgets import Slider
        #
        # downscale_ratio = 5
        # image_dn = downscale_local_mean(img_as_ubyte(image), (downscale_ratio, downscale_ratio))
        #
        # centers = table[['centroid_x', 'centroid_y']].values // downscale_ratio
        # pos_cells_old = np.where(meanInt > thresh)[0]
        #
        # ###############################################################################
        # fig = plt.figure()
        # plt.axis('off')
        # plt.title(bioM)
        # img_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
        # slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
        #
        # plt.axes(img_ax)
        # plt.axis('off')
        # plt.imshow(image_dn, cmap='gray')
        # dot_plt, = plt.plot(centers[pos_cells_old, 0], centers[pos_cells_old, 1], 'r.')
        # a_slider = Slider(slider_ax, 'Thresh', 0, np.max(meanInt), valinit=thresh)
        #
        # def update(val):
        #     print(val)
        #     val = a_slider.val
        #     pos_samples = np.where(meanInt > val)[0]
        #     dot_plt.set_xdata(centers[pos_samples, 0])
        #     dot_plt.set_ydata(centers[pos_samples, 1])
        #     fig.canvas.draw()
        #
        # a_slider.on_changed(update)
        #
        # plt.show()
        #
        # a = 1

    table.to_csv(os.path.join(save_dir, 'classification_table.csv'))


def generate_boolean_center_image_from_sheet():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get image info
    image = imread_collection(image_sample, plugin='tifffile')
    im_size = image[0].shape[::-1]

    # read bbxs file
    assert os.path.isfile(classification_table_full), '{} not found!'.format(classification_table_base)
    # if file exist -> load
    class_table = pd.read_csv(classification_table_full, sep=',')
    class_table.set_index('ID', inplace=True)

    # # ICE file requires different format, restore to original format
    # centers = class_table[['centroid_x', 'centroid_y']].values
    # centers[:, 1] = im_size[1] - centers[:, 1]

    # read boolean file
    assert os.path.isfile(boolean_table), '{} not found!'.format(boolean_table)
    # if file exist -> load
    bool_table = pd.read_excel(boolean_table, sheet_name=None)

    # create table to write cell types
    cell_type_table = class_table.copy()[['centroid_x', 'centroid_y']]
    cell_type_table['cell_type'] = ''

    for sheet in bool_table.items():
        name, table = sheet
        table = table.replace('+', 1.0)
        table = table.replace('-', 0.0)
        table.set_index('Type', inplace=True)

        # for each cell type:
        all_cells = []
        for index, cell_type in table.iterrows():

            # remove +/- from the list because we don't care
            # cell_type = cell_type[cell_type != 'Nan']
            cell_type = cell_type[~cell_type.isnull()]

            # find rows in classification table that match the cell_type row
            pos_cells = np.where((class_table[cell_type.keys()] == cell_type.values).all(1))[0]

            # temp
            # all_cells.extend(pos_cells)

            # update the cell type table with the cell type class
            cell_type_table.loc[pos_cells + 1, 'cell_type'] = cell_type.name

            # find centers corresponding to rows
            centers = class_table.loc[pos_cells + 1, ['centroid_x', 'centroid_y']].values
            all_cells.append(centers.shape[0])

            # # temp
            # import matplotlib.pyplot as plt
            # from skimage.transform import rescale
            # downscale_ratio = 3
            # image = np.zeros((29398, 43054, 3), dtype=np.uint16)
            # image[:, :, 0] = imread(os.path.join(image_dir, 'S1_R3C10.tif'), plugin='tifffile')
            # image[:, :, 1] = imread(os.path.join(image_dir, 'S1_R4C5.tif'), plugin='tifffile')
            # image[:, :, 2] = imread(os.path.join(image_dir, 'S1_R3C5.tif'), plugin='tifffile')
            # image = img_as_ubyte(image)
            # image_dn = rescale(image, (1 / downscale_ratio, 1 / downscale_ratio), preserve_range=True,
            #                    multichannel=True).astype(np.uint8)
            # plt.figure()
            # plt.imshow(image_dn)
            # # centerts = class_table.loc[class_table['NeuN'] == 1, ['centroid_x', 'centroid_y']].values
            # # centers = class_table.loc[pos_cells + 1, ['centroid_x', 'centroid_y']].values
            # plt.scatter(centers[:, 0] // downscale_ratio, centers[:, 1] // downscale_ratio, s=1)
            # plt.title(index)

            # save the image
            # center_image(os.path.join(save_dir, cell_type.name + '.tif'), centers, im_size, r=center_size, color='red')

    # for paper
    cell_type.name = 'Catecholaminergic neuron'
    cntrs = cell_type_table.loc[cell_type_table['cell_type'] == cell_type.name, ['centroid_x', 'centroid_y']].values
    center_image(os.path.join(save_dir, cell_type.name + '.tif'), cntrs, im_size, r=center_size, color='red')



    uncategorized_cells = cell_type_table.loc[cell_type_table['cell_type'] == '', ['centroid_x', 'centroid_y']].values
    center_image(os.path.join(save_dir, 'uncategorized.tif'), uncategorized_cells, im_size, r=center_size)
    print((np.sum(cell_type_table['cell_type'] == '') / 219634) * 100)
    cell_type_table.to_csv(os.path.join(save_dir, 'cell_type_table.csv'))


if __name__ == '__main__':

    start = time.time()

    # generate_centers_image()
    # generate_classification_table()
    # generate_boolean_center_image()
    # generate_boolean_center_image_2()

    update_classification_table()
    # generate_boolean_center_image_from_sheet()

    # generate_boolean_image()
    print('*' * 50)
    print('*' * 50)
    print('Pipeline finished successfully in {} seconds.'.format(time.time() - start))
