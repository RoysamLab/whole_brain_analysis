import os
import argparse
import tensorflow as tf
import numpy as np
import skimage, skimage.io
import progressbar

from lib.ops import check_path
from lib.image_uitls import *


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='data/test', help='path to the large input images direcotry')
parser.add_argument('--crop_size', type=str, default='2000,2000', help='size of the cropped image e.g. 300,300')
parser.add_argument('--model_dir', type=str, default='freezed_models/NeuN_2000', help='path to exported model directory')
parser.add_argument('--output_file', type=str, default='data/test/bbxs.txt', help='path to txt file of coordinates')
parser.add_argument('--score_threshold', type=int, default=.5, help='threshold on keeping the bbxs ')
parser.add_argument('--visualize_crop', type=int, default=6, help='visualize n sample images with bbxs')
parser.add_argument('--adjust_image', action='store_true',  help='adjust histogram of image for visualization')


args = parser.parse_args()

def get_crop(image_filenames, crop_size=[300, 300], adjust_hist=False):

    # crop width and height
    crop_width, crop_height = crop_size

    # read images
    image = read_image_from_filenames(image_filenames)

    # get image information
    img_rows, img_cols, img_ch = image.shape  # img_rows = height , img_cols = width

    # overlap between crops
    ovrlp = 50

    max_bar = (img_rows // (crop_height - ovrlp) + 1) * (img_cols // (crop_width - ovrlp) +1)
    crop_count = 0
    with progressbar.ProgressBar(max_value=max_bar) as bar:
        # get each crop
        for i in range(0, img_rows, crop_height - ovrlp):
            for j in range(0, img_cols, crop_width - ovrlp):
                bar.update(crop_count)
                # crop the image
                crop_img = image[i:i + crop_height, j:j + crop_width, :]   # create crop image

                # if we were at the edges of the image, zero pad the crop
                if crop_img.shape[:2][::-1] != crop_size:
                    temp = np.copy(crop_img)
                    crop_img = np.zeros((crop_height, crop_width, crop_img.shape[2]))
                    crop_img[:temp.shape[0], :temp.shape[1], :] = temp

                crop_count = crop_count + 1

                yield [j, i], crop_img


def load_graph(graph, ckpt_path):
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


def get_bbxs(sess, image, boxes=None, classes=None, scores=None, image_tensor=None):
    boxes_out, classes_out, scores_out = sess.run([boxes, classes, scores],
                                                  feed_dict={image_tensor: image})
    boxes_out = boxes_out.squeeze()
    classes_out = classes_out.squeeze().astype(np.int32)
    scores_out = scores_out.squeeze()
    return boxes_out, classes_out, scores_out


def get_features(sess, image, features_tensor=None, image_tensor=None):
    feat_avg_out = sess.run(features_tensor, feed_dict={image_tensor: image})
    return feat_avg_out.squeeze()


def non_max_suppression_fast(boxes, overlapThresh):
    # Malisiewicz et al.
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

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def main():

    # read input
    input_fnames = []
    for file in os.listdir(args.input_dir):
        file_name, file_extension = os.path.splitext(file)
        if file_extension in ['.jpeg', '.jpg', '.bmp', '.tif', '.tiff', '.png']:
            input_fnames.append(check_path(os.path.join(args.input_dir, file)))
    assert len(input_fnames) <= 3, ('Provide no more than 3 images')

    # read crop size
    crop_size = list(map(int, args.crop_size.split(',')))

    # read Boolean to adjust the image histogram
    adjust_image = args.adjust_image

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = check_path(args.model_dir + '/frozen_inference_graph.pb')

    # load frozen model to memory
    detection_graph = tf.Graph()
    load_graph(detection_graph, PATH_TO_CKPT)

    # get needed tensors
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')     # (1, ?, ?, 3)
    boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')      # (?, 4)
    scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')    # (?,)
    classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')  # (?,)
    features_tensor = detection_graph.get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')  # (?, 1, 1, 1536)
    # feat_conv = detection_graph.get_tensor_by_name('SecondStageFeatureExtractor/InceptionResnetV2/Conv2d_7b_1x1/Relu:0')   # (?, 8, 8, 1536)

    # Load tf model into memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config, graph=detection_graph)


    # for each crop:
    bbxs = []
    features = []
    crop_idx = 0

    for corner, crop in get_crop(input_fnames, crop_size=crop_size):
        crop = np.expand_dims(crop, 0)
        # predict bbxs
        boxes, classes, scores = get_bbxs(sess, crop, boxes=boxes_tensor, classes=classes_tensor,
                                          scores=scores_tensor, image_tensor=image_tensor)
        # get features of bbxs
        crop_features = get_features(sess, crop, features_tensor=features_tensor, image_tensor=image_tensor)

        # keep the predictions higher than threshold
        keep_boxes = scores > args.score_threshold
        # if no box found in the crop, continue
        if not np.any(keep_boxes):
            continue

        # keep boxes higher than the threshold
        crop_features = crop_features[keep_boxes, :]
        boxes = boxes[keep_boxes, :]

        crop_bbxs = []
        for i, box in enumerate(boxes):
            box = box.tolist()
            ymin, xmin, ymax, xmax = box

            # for crop visualization
            crop_bbxs.append([xmin * crop_size[0],
                              ymin * crop_size[1],
                              xmax * crop_size[0],
                              ymax * crop_size[1]])

            xmin = np.rint(xmin * crop_size[0] + corner[0]).astype(int)
            xmax = np.rint(xmax * crop_size[0] + corner[0]).astype(int)
            ymin = np.rint(ymin * crop_size[1] + corner[1]).astype(int)
            ymax = np.rint(ymax * crop_size[1] + corner[1]).astype(int)


            # append each bbx to the list
            bbxs.append([xmin, ymin, xmax, ymax])
            # append the features
            features.append(crop_features[i, :])


        # visualize bbxs
        if crop_idx < args.visualize_crop:
            visualize_bbxs(np.squeeze(crop), bbxs=np.array(crop_bbxs))
        crop_idx = crop_idx + 1

    # close the session after done!
    sess.close()

    # remove overlapping bounding boxes due to cropping with overlap
    # bbxs = non_max_suppression_fast(np.array(bbxs), .5)

    # get centers
    centers = np.empty((bbxs.shape[0], 2), dtype=int)
    centers[:, 0] = np.rint((bbxs[:, 0] + bbxs[:, 2]) / 2).astype(int)
    centers[:, 1] = np.rint((bbxs[:, 1] + bbxs[:, 3]) / 2).astype(int)

    # create a column for unique IDs
    ids = np.arange(1, bbxs.shape[0] + 1)

    # create numpy array for the table
    table = np.hstack((ids, centers, bbxs))

    fmt = '\t'.join(['%d'] * table.shape[1])
    hdr = '\t'.join(
        ['ID'] + ['centroid_x'] + ['centroid_y'] + ['xmin'] + ['ymin'] + ['xmax'] + ['ymax'] +
        ['f{}'.format(i) for i in range(1, crop_features.shape[1])])
    cmts = ''

    # check if output dir exist
    if not os.path.isdir(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    # save table to output file
    np.savetxt(args.output_file, table, fmt=fmt, header=hdr, comments=cmts)




if __name__ == '__main__':

    main()
    print()
