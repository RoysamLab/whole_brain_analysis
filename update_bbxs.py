import os
import numpy as np
import xml.etree.ElementTree as ET
import argparse
from lib.ops import non_max_suppression_fast
from lib.ops import check_path


def MergeUpdatedBboxs(xml_dir, output_file):

    bbxs = []

    for fname in os.listdir(xml_dir):
        # get the x, y coordinates of the corner of the crop
        y, x = map(int, os.path.basename(fname).split('.')[0].split('_'))

        # read file
        tree = ET.parse(os.path.join(xml_dir, fname))

        for i, Obj in enumerate(tree.findall('object')):  # take the current animal
            bndbox = Obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            bbxs.append([xmin + x, ymin + y, xmax + x, ymax + y])

    # TBD
    # remove overlapping bounding boxes due to cropping with overlap
    # bbxs = non_max_suppression_fast(np.array(bbxs), .5)

    # get centers
    centers = np.empty((bbxs.shape[0], 2), dtype=int)
    centers[:, 0] = np.rint((bbxs[:, 0] + bbxs[:, 2]) / 2).astype(int)
    centers[:, 1] = np.rint((bbxs[:, 1] + bbxs[:, 3]) / 2).astype(int)

    # create a column with IDs
    ids = np.expand_dims(np.arange(1, np.shape(centers)[0]+1), axis=1)

    # make numpy array for the table
    table = np.hstack((ids, centers, bbxs))

    # save to the file
    np.savetxt(output_file, table,  comments='',
               fmt='%d\t%d\t%d\t%d\t%d\t%d\t%d',
               header='ID\tcentroid_x\tcentroid_y\txmin\tymin\txmax\tymax')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_dir', type=str, default='data/train/xmls', help='path to the directory of xml files')
    parser.add_argument('--output_file', type=str, default='data/train/input_data/bbxs.txt', help='path to the output file name')
    args = parser.parse_args()

    xml_dir = check_path(args.xml_dir)
    save_fname = check_path(args.output_file)

    MergeUpdatedBboxs(xml_dir, save_fname)
