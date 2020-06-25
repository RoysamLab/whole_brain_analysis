import os
import tqdm
import progressbar
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import spatial
from skimage.io import imread, imsave
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import concurrent.futures


def pixel_to_micro_square(area_pixel, um_per_pixel=0.325):
    """
    Transform area from px*px to um*um based on px/um ratio
    :param area_pixel: area in pixel
    :param um_per_pixel: px/um ratio
    :return:
    """
    return area_pixel * (um_per_pixel**2)


def count_per_normalized_area(count, area, normalized_area=10**6, **kwargs):
    """
    Count the cells of reagion in a normalized um^2 area (density per normalized area)
    :param count: count of cells in the region
    :param area: area of reagion in px^2
    :param normalized_area: normalized area in um^2
    :param um_per_pixel: px/um ratio
    :return:
    """
    return (count * normalized_area) / (pixel_to_micro_square(area,  **kwargs))


def ismember_row(a, b):
    """
    find index of rows in a that is a member in b
    """

    def asvoid(arr):
        """
        View the array as dtype np.void (bytes)
        This views the last axis of ND-arrays as bytes so you can perform comparisons on
        the entire row.
        http://stackoverflow.com/a/16840350/190597 (Jaime, 2013-05)
        Warning: When using asvoid for comparison, note that float zeros may compare UNEQUALLY
        >>> asvoid([-0.]) == asvoid([0.])
        array([False], dtype=bool)
        """
        arr = np.ascontiguousarray(arr)
        return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

    voida, voidb = map(asvoid, (a, b))
    return np.where(np.in1d(voidb, voida))[0]


def generate_regions_table(table):
    # create dataframe for regions and number of cell types in region for heatmap
    columns = [column for column in table.columns if column not in ['ID', 'centroid_x', 'centroid_y',
                                                                    'xmin', 'ymin', 'xmax', 'ymax', 'region']]
    indices = [region for region in table['region'].unique() if region is not '']
    heatmap_table = pd.DataFrame(columns=columns, index=indices)
    heatmap_table.index.name = 'region'

    for region_name, row in heatmap_table.iterrows():
        heatmap_row = table.loc[table['region'] == region_name, :].sum(axis=0)
        heatmap_row.drop(['centroid_x', 'centroid_y', 'xmin', 'ymin', 'xmax', 'ymax', 'region'], inplace=True)
        heatmap_row.name = region_name

        heatmap_table.loc[heatmap_row.name] = heatmap_row

    heatmap_table = heatmap_table[heatmap_table.columns].astype(int)
    return heatmap_table


def add_region_to_classification_table(table_filename, images_path, output_dir, label_all_cells=False):
    # create dir to save images if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read the classification table
    table = pd.read_csv(table_filename, sep=',')
    table.set_index('ID', inplace=True)

    # add empty column for region
    table['region'] = ''

    # list all image files in the folder
    image_files = [f for f in os.listdir(images_path) if '.tif' in f]

    # get all points in array format
    all_points = table.loc[:, ['centroid_x', 'centroid_y']].values

    bar = progressbar.ProgressBar(max_value=len(image_files))

    # make a list of tuples with regions
    region_area_list = []

    # # for each region
    # def process_region(image_fname):
    #     # get region name
    #     region_name, _ = os.path.splitext(image_fname)
    #
    #     # create binary image with region pixels
    #     region_mask = imread(os.path.join(images_path, image_fname), plugin='tifffile')
    #     region_mask[region_mask != 0] = 1
    #
    #     # create binary image with all centers
    #     centers_mask = np.zeros_like(region_mask)
    #     centers_mask[table['centroid_y'], table['centroid_x']] = 1
    #
    #     # multiplying region mask and centers mask will give the centers inside the region
    #     region_centers_mask = np.multiply(region_mask, centers_mask)
    #
    #     # convert image to array
    #     [center_y, center_x] = np.nonzero(region_centers_mask)
    #     region_points = np.array([center_x, center_y]).transpose()
    #
    #     # find the index of points in the table
    #     # python starts with 0 but ID (index) starts with 1 -> +1
    #     region_idx = ismember_row(region_points, all_points)
    #     region_idx += 1
    #
    #     # update the region column with the name of the region
    #     table.loc[region_idx, 'region'] = region_name
    #
    #     # save number of pixels in each region in another file
    #     region_area_list.append([region_name, np.count_nonzero(region_mask)])
    #
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     tqdm(executor.map(process_region, image_files), total=len(image_files))

    for id, image_fname in enumerate(image_files):
        # get region name
        region_name, _ = os.path.splitext(image_fname)

        # create binary image with region pixels
        region_mask = imread(os.path.join(images_path, image_fname), plugin='tifffile')
        region_mask[region_mask != 0] = 1

        # create binary image with all centers
        centers_mask = np.zeros_like(region_mask)
        centers_mask[table['centroid_y'], table['centroid_x']] = 1

        # multiplying region mask and centers mask will give the centers inside the region
        region_centers_mask = np.multiply(region_mask, centers_mask)

        # convert image to array
        [center_y, center_x] = np.nonzero(region_centers_mask)
        region_points = np.array([center_x, center_y]).transpose()

        # find the index of points in the table
        # python starts with 0 but ID (index) starts with 1 -> +1
        region_idx = ismember_row(region_points, all_points)
        region_idx += 1

        # update the region column with the name of the region
        table.loc[region_idx, 'region'] = region_name

        # # check
        # plt.figure()
        # plt.imshow(region_mask, cmap='gray')
        # plt.plot(table.loc[table.region == region_name, 'centroid_x'],
        #          table.loc[table.region == region_name, 'centroid_y'], 'r.')
        # plt.plot()

        # save number of pixels in each region in another file
        region_area_list.append([region_name, np.count_nonzero(region_mask)])

        # update bar
        bar.update(id)

    if label_all_cells:
        # for cells with no regions -> find closest cell and assign same region
        with_region_points = table[~(table['region'] == '')].reset_index()
        no_region_points = table[table['region'] == ''].reset_index()

        tree = spatial.KDTree(with_region_points.loc[:, ['centroid_x', 'centroid_y']].values)
        _, closest_idx = tree.query(no_region_points.loc[:, ['centroid_x', 'centroid_y']].values)
        table.loc[table['region'] == '', 'region'] = with_region_points.loc[closest_idx, 'region'].values

        # # check
        # image_size = (29398, 43054)
        # t1 = no_region_points.loc[:, ['centroid_x', 'centroid_y']].values
        # x1 = t1[:, 0]
        # y1 = t1[:, 1]
        # y1 = image_size[0] - y1
        #
        # t2 = with_region_points.loc[closest_idx, ['centroid_x', 'centroid_y']].values
        # x2 = t2[:, 0]
        # y2 = t2[:, 1]
        # y2 = image_size[0] - y2
        #
        # plt.figure()
        # plt.plot(image_size[1], image_size[0], 'k.')
        # plt.plot(x1, y1, 'r.')
        # plt.plot(x2, y2, 'b.')
        # plt.show()

    # save region info as a table
    regions_table = generate_regions_table(table)
    regions_table.insert(0, 'Area(px)', '')
    regions_table.insert(1, 'Area(um)', '')
    for region in region_area_list:
        regions_table.loc[region[0], 'Area(px)'] = region[1]
        regions_table.loc[region[0], 'Area(um)'] = pixel_to_micro_square(region[1])
    regions_table['Area(um)'] = regions_table['Area(um)'].astype(int)
    regions_table.to_csv(os.path.join(output_dir, 'atlas_regions_table.csv'))

    # save updated classification table with regions in output directory
    save_name = os.path.join(output_dir, 'regions_table.csv')
    table.to_csv(save_name)


def plot_table_heatmap(region_table):
    table = pd.read_csv(region_table, sep=',')
    table.set_index('ID', inplace=True)

    heatmap_table = generate_regions_table(table)

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111)
    sns.heatmap(heatmap_table.transpose(), vmin=0, vmax=heatmap_table.max().max(), robust=True, ax=ax,
                cbar_kws={'label': 'Number of cells'})
    ax.set_ylabel('Phenotypes')
    ax.set_xlabel('Regions')
    plt.tight_layout()
    plt.savefig('heatmap.png', dpi=300)

    plt.show()

    heatmap_table_red = heatmap_table[heatmap_table.index.str.contains("L-CX")]
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111)
    sns.heatmap(heatmap_table_red.transpose(), vmin=0, vmax=heatmap_table_red.max().max(), robust=True, ax=ax,
                annot=True, fmt='g',
                cbar_kws={'label': 'Number of cells'})
    ax.set_ylabel('Phenotypes')
    ax.set_xlabel('Regions')
    plt.tight_layout()
    plt.savefig('heatmap.png', dpi=300)

    plt.show()


def plot_atlas_heatmap(images_path, atlas_table_fname, biomarker='NeuN'):

    # read the atlas reagion table
    atlas_table = pd.read_csv(atlas_table_fname, sep=',')
    atlas_table.set_index('region', inplace=True)

    # create a black image to build the heatmap on top of
    sample_image = [f for f in os.listdir(images_path) if '.tif' in f][0]
    image = imread(os.path.join(images_path, sample_image))
    image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # calculate the density of channel per region for color map -> get min and max for range
    densities = np.array([count_per_normalized_area(r[biomarker], r['Area(px)']) for _, r in atlas_table.iterrows()])
    cmap_min = np.min(densities)
    cmap_max = np.max(densities)
    norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)
    cmap = cm.rainbow
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    bar = progressbar.ProgressBar(max_value=atlas_table.shape[0])

    # for each region color the binary image with coded color from colormap
    for idx, (region, row) in enumerate(atlas_table.iterrows()):

        region_mask = imread(os.path.join(images_path, region+'.tif'))
        region_mask[region_mask != 0] = 1

        region_density = count_per_normalized_area(row[biomarker], row['Area(px)'])
        region_color = np.multiply(m.to_rgba(region_density), 255).astype(int)

        for i in range(3):
            image[:, :, i] += region_mask * region_color[i]

        bar.update(idx)

    imsave(biomarker + '.tif', image)

if __name__ == '__main__':
    class_table = r'E:\50_plex\tif\pipeline2\classification_results\classification_table.csv'
    images_path = r'E:\50_plex\temp'
    output_dir = r'E:\50_plex\temp'
    label_all_cells = False
    add_region_to_classification_table(class_table, images_path, output_dir, label_all_cells=label_all_cells)

    # atlas_table = r'E:\50_plex\tif\pipeline2\classification_results\regions\atlas_regions_table.csv'
    # for biom in ['NeuN', 'S100', 'Olig2', 'Iba1', 'RECA1']:
    #     plot_atlas_heatmap(images_path, atlas_table, biomarker=biom)
    # # plot_table_heatmap(region_table)
