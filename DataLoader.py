import os
import pandas as pd
import random
import scipy
import numpy as np
import scipy.spatial as spatial
import warnings
import skimage
from skimage import exposure
import xml.etree.ElementTree as ET
import itertools
from lib.image_uitls import read_image_from_filenames, visualize_bbxs
from lib.segmentation import GenerateBBoxfromSeeds
from lib.ops import write_xml
import progressbar


class DataLoader(object):

    def __init__(self, config):

        self.config = config

        # read all images
        image_filenames = []
        for i in range(config.channel):
            filename = getattr(config, 'c{}'.format(i+1))
            if filename is not None:
                image_filenames.append(os.path.join(config.ipnut_dir, filename))

        if len(image_filenames) != 0:
            self.image = read_image_from_filenames(image_filenames, to_ubyte=False)
        else:
            self.image = None

        self.height = config.height
        self.width = config.width
        self.channel = config.channel
        self.overlap = config.overlap

        # read centers if exist
        if config.centers_file is not None:
            centers_fname = os.path.join(os.path.join(config.input_dir, config.centers_file))
            assert os.path.isfile(centers_fname), '{} not found!'.format(centers_fname)
            # if file exist -> load
            centers_table = pd.read_csv(centers_fname, sep='\t')
            self.centers = centers_table[['centriod_x', 'centriod_y']].values
        else:
            self._centers = None

        # read bbxs if exist
        if config.bbxs_file is not None:
            bbxs_fname = os.path.join(os.path.join(config.input_dir, config.bbxs_file))
            assert os.path.isfile(bbxs_fname), '{} not found!'.format(bbxs_fname)
            # if file exist -> load
            bbxs_table = pd.read_csv(bbxs_fname, sep='\t')
            self.bbxs = bbxs_table[['xmin', 'ymin', 'xmax', 'ymax']].values
        else:
            self._bbxs = None

        self._scores = None

    @property
    def centers(self):
        return self._centers

    @staticmethod
    def get_centers(bbxs):
        centers = np.empty((bbxs.shape[0], 2), dtype=int)
        centers[:, 0] = (bbxs[:, 0] + bbxs[:, 2]) // 2
        centers[:, 1] = (bbxs[:, 1] + bbxs[:, 3]) // 2
        return centers

    @property
    def bbxs(self):
        return self._bbxs

    @bbxs.setter
    def bbxs(self, value):
        self._bbxs = value
        self._centers = self.get_centers(value)

    def save_bbxs(self, filename):
        # create a column for unique IDs
        ids = np.expand_dims(np.arange(1, self._bbxs.shape[0] + 1), axis=1)

        # create numpy array for the table
        table = np.hstack((ids, self._centers, self._bbxs))

        fmt = '\t'.join(['%d'] * table.shape[1])
        hdr = '\t'.join(['ID'] + ['centroid_x'] + ['centroid_y'] + ['xmin'] + ['ymin'] + ['xmax'] + ['ymax'])
        cmts = ''

        np.savetxt(filename, table, fmt=fmt, header=hdr, comments=cmts)

    @property
    def scores(self):
        return self._scores

    @scores.setter
    def scores(self, value):
        self._scores = value

    def next_crop(self, crop_width=None, crop_height=None, crop_overlap=None):
        '''
        Generator for cropping large image based on given width, height and overlab
        :param crop_width: width of crop
        :param crop_height: height of crop
        :param crop_overlap: overlap between crops
        :yeild top left corner coordinates [x, y], cropped image
        '''
        if crop_width is None:
            crop_width = self.width
        if crop_height is None:
            crop_height = self.height
        if crop_overlap is None:
            crop_overlap = self.overlap

        # get image information
        img_rows, img_cols, img_ch = self.image.shape  # img_rows = height , img_cols = width
        max_bar = (img_rows // (crop_height - crop_overlap) + 1) * (img_cols // (crop_width - crop_overlap) + 1)

        bar = progressbar.ProgressBar(max_value=max_bar)

        bar.start()
        bar_idx = 1
        # get each crop
        for i in range(0, img_rows, crop_height - crop_overlap):
            for j in range(0, img_cols, crop_width - crop_overlap):

                # update bar
                bar.update(bar_idx)

                # temporary store the values of crop
                temp = self.image[i:i + crop_height, j:j + crop_width, :]

                # create new array to copy temporary stored values
                crop_img = np.zeros((crop_height, crop_width, self.image.shape[-1]), dtype=self.image.dtype)
                crop_img[:temp.shape[0], :temp.shape[1], :] = temp

                yield [j, i], crop_img
                bar_idx += 1
        bar.finish()

    def get_bbxs_from_image(self, image, corner):
        '''
        crop from self.bbxs or generate bounding boxes for given image
        :param image: image
        :param corner: top left corner coordinates [x, y]
        :retrun : np.array of found bounding boxes , np.array of truncation of objects
        '''
        [j, i] = corner
        [image_width, image_height] = image.shape[:2][::-1]

        # at this point we can only extract bbxs provided the centers
        # TODO: add section fully automatic detection
        # find cells that center is in the crop
        idx = np.where(((self.centers[:, 0] > j) &
                        (self.centers[:, 0] < j + image_width) &
                        (self.centers[:, 1] > i) &
                        (self.centers[:, 1] < i + image_height)),
                       True, False)

        # if user provides the bounding boxes, extract them
        if self._bbxs is not None:
            if not np.any(idx):  # if no cell in the crop, SKIP
                return None, None

            # extract bbxs in the crop
            crop_bbxs = self.bbxs[idx, :]
            # shift the x & y values based on crop size
            crop_bbxs[:, [0, 2]] = crop_bbxs[:, [0, 2]] - j
            crop_bbxs[:, [1, 3]] = crop_bbxs[:, [1, 3]] - i

        # if user provides the centers, run segmentation to extract bounding boxes
        elif self._centers is not None:
            crop_centers = self.centers[idx, :]

            # shift the x & y values based on crop size
            crop_centers[:, 0] = crop_centers[:, 0] - j
            crop_centers[:, 1] = crop_centers[:, 1] - i

            # generate bounding boxes using segmentation
            dapi = np.copy(image[:, :, 0])
            # for 16bit images only.
            # TODO: general form for all types
            dapi = exposure.rescale_intensity((dapi // 256).astype('uint8'),
                                              in_range='image', out_range='dtype')
            crop_bbxs = GenerateBBoxfromSeeds(dapi, crop_centers)

        # find truncated objects in crop
        crop_truncated = np.where(((crop_bbxs[:, 0] < 0) |
                                   (crop_bbxs[:, 1] < 0) |
                                   (crop_bbxs[:, 2] > image_width) |
                                   (crop_bbxs[:, 3] > image_height)), True, False)
        # clip truncated objects
        if np.any(crop_truncated):
            crop_bbxs[:, [0, 2]] = np.clip(crop_bbxs[:, [0, 2]], 1, image_width - 1)
            crop_bbxs[:, [1, 3]] = np.clip(crop_bbxs[:, [1, 3]], 1, image_height - 1)

        return crop_bbxs, crop_truncated

    @staticmethod
    def remove_close_centers(centers, scores=None, radius=3):
        '''
        find close centers and remove centers with radius < radius
        :param centers: np.array of [centroid_x, centroid_y]
        :param scores: np.array of probability of objectness
        :param radius: int of distance threshold between objects
        :return: array of True and Flase for centers to keep(True) remove(False)
        '''

        # get groups of centers
        tree = spatial.cKDTree(centers)
        groups = tree.query_ball_point(centers, radius)

        # remove isolated centers
        groups = [group for group in groups if len(group) > 1]

        # if no groups, return all True
        if len(groups) == 0:
            return np.array([True] * centers.shape[0])

        # remove duplicated groups
        groups.sort()
        groups = list(groups for groups,_ in itertools.groupby(groups))

        # remove the center with highest probability and add to to_be_removed list
        to_be_removed = []
        for i, group in enumerate(groups):
            if scores is not None:
                max_idx = np.argmax(scores[group])
            else:
                # if we don't have the scores remove the first object
                max_idx = 0
            to_be_removed.append(np.delete(groups[i], max_idx))

        # find index of centers to be removed
        to_be_removed = np.unique(list(itertools.chain.from_iterable(to_be_removed)))

        # return invert of to_be_removed = to_keep
        return np.isin(np.arange(centers.shape[0]), to_be_removed, invert=True)

    @staticmethod
    def nms(boxes, overlapThresh=.7):
        '''

        :param boxes: np.array of [xmin, ymin, xmax, ymax]
        :param overlapThresh: float intersection over union threshold
        :return: array of True and Flase for centers to keep(True) remove(False)
        '''
        # non_max_suppression_fast Malisiewicz et al.
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return pick

    def write_crops(self, save_folder, crop_width=None, crop_height=None, crop_overlap=None, adjust_hist=False):
        '''

        :param save_folder: folder path to save imgs and xmls folders
        :param crop_width: width of crop
        :param crop_height: height of crop
        :param crop_overlap: overlap between crop
        :param adjust_hist: boolean to adjust the histogram of image (rescale the intensity)
        :return:
        '''

        if crop_width is None:
            crop_width = self.width
        if crop_height is None:
            crop_height = self.height
        if crop_overlap is None:
            crop_overlap = self.overlap

        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        # check for subdirectories
        dir_list = os.listdir(save_folder)
        if 'imgs' not in dir_list:
            os.mkdir(os.path.join(save_folder, 'imgs'))
        if 'xmls' not in dir_list:
            os.mkdir(os.path.join(save_folder, 'xmls'))

        crop_gen = self.next_crop(crop_width=crop_width, crop_height=crop_height, crop_overlap=crop_overlap)
        idx = 1
        while True:
            try:
                [j, i], crop_image = next(crop_gen)
            except StopIteration:
                break

            crop_bbxs, crop_truncated = self.get_bbxs_from_image(crop_image, [j, i])

            if crop_bbxs is None:
                continue

            # save image and xml:
            filename = '{:05d}_{:05d}'.format(j, i)  # save name as x_y format of top left corner of crop
            if adjust_hist:
                # for 16bit images only.
                # TODO: general form for all types
                crop_image = exposure.rescale_intensity((crop_image // 256).astype('uint8'),
                                                        in_range='image', out_range='dtype')
            # save to folder
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skimage.io.imsave(os.path.join(save_folder, 'imgs', filename + '.jpeg'),
                                  np.squeeze(crop_image))  # save the image

            # write bounding boxes in xml file
            labels = ['Nucleus'] * crop_bbxs.shape[0]
            truncated = crop_truncated * 1
            write_xml(os.path.join(save_folder, 'xmls', filename + '.xml'), corner=[j, i],
                      bboxes=crop_bbxs, labels=labels, truncated=truncated,
                      image_size=[crop_width, crop_height, self.channel])

            # visualize_bbxs(crop_image, bbxs=crop_bbxs, centers=crop_centers)
            idx += 1

        print('{} images in: {}'.format(idx - 1, save_folder + '/imgs'))
        print('{} xmls in: {}'.format(idx - 1, save_folder + '/xmls'))

    def update_xmls(self, xml_dir, save_fname):
        '''

        :param xml_dir: path to the folder containing xml files
        :param save_fname: save new bbxs to txt file
        :return:
        '''

        to_be_deleted = []
        to_be_added = []
        for filename in os.listdir(xml_dir):

            # read file
            tree = ET.parse(os.path.join(xml_dir, filename))
            #TODO: check [::-1] for old crops...
            corner = list(map(int, os.path.basename(filename).split('.')[0].split('_')))

            size = tree.find('size')
            crop_width = int(size.find('width').text)
            crop_height = int(size.find('height').text)

            # find objects in center (not close to the edge) of crop to be deleted from self.bbxs
            to_be_deleted_crop = np.where((self._bbxs[:, 0] > corner[0] + 50) &
                                          (self._bbxs[:, 1] > corner[1] + 50) &
                                          (self._bbxs[:, 2] < corner[0] + crop_width - 50) &
                                          (self._bbxs[:, 3] < corner[1] + crop_height - 50))[0]
            to_be_deleted.append(to_be_deleted_crop) if len(to_be_deleted_crop) > 0 else None

            # extract bbxs from xml file
            for i, Obj in enumerate(tree.findall('object')):  # take the current animal
                bndbox = Obj.find('bndbox')
                box = np.array([int(bndbox.find('xmin').text),
                                int(bndbox.find('ymin').text),
                                int(bndbox.find('xmax').text),
                                int(bndbox.find('ymax').text)])

                # if box was inside the crop (not close to the edge) to be added to self.bbxs
                if box[0] >= 50 and box[1] >= 50 and box[2] <= crop_width - 50 and box[3] <= crop_height - 50:
                    box = box + np.array([corner[0], corner[1], corner[0], corner[1]])
                    to_be_added.append(box)

        # update the bbxs
        to_be_deleted = [item for sublist in to_be_deleted for item in sublist]
        self.bbxs = np.delete(self.bbxs, to_be_deleted, axis=0)
        self.bbxs = np.vstack([self._bbxs, to_be_added])
        self.bbxs = np.unique(self._bbxs, axis=0)

        self.save_bbxs(os.path.join(self.config.input_dir, save_fname))

        from lib.image_uitls import bbxs_image, center_image
        bbxs_image(os.path.join(self.config.input_dir, 'updated_bbxs.tif'), self.bbxs, self.image.shape[:2][::-1])
        center_image(os.path.join(self.config.input_dir, 'centers_validation.tif'), self.centers, self.image.shape[:2][::-1])
        print('{} updated with new objects in {}'.format(self.config.bbxs_file, xml_dir))
        print('new bbxs saved in {}'.format(os.path.join(self.config.input_dir, save_fname)))

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        shuffled_x = self.x_train[permutation, :, :, :]
        shuffled_y = self.y_train[permutation]
        return shuffled_x, shuffled_y
