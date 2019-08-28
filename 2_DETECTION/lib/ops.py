import sys
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


def check_path(loc):
    if sys.platform == "linux" or sys.platform == "linux2":
        locLinux = loc.replace("\\", "/")
        return locLinux
    if sys.platform == "win32" or sys.platform == "win64":
        locWin = loc.replace("/", '\\')
        return locWin
    else:
        return loc


def write_object(f, xmin, ymin, xmax, ymax, label, trunc):
    '''
    Write object information in xml file
    :param f: writer object
    :param xmin: top left corner x
    :param ymin: top left corner y
    :param xmax: bottom right corner x
    :param ymax: bottom right corner y
    :param label: string name of the class
    :param trunc: status of truncated or not truncated object
    :return: None
    '''
    line = '\t<object>\n'
    f.writelines(line)

    line = '\t\t<name>' + label +'</name>\n'
    f.writelines(line)

    line = '\t\t<pose>Unspecified</pose>\n'
    f.writelines(line)

    line = '\t\t<truncated>' + trunc + '</truncated>\n'
    f.writelines(line)

    line = '\t\t<difficult>0</difficult>\n'
    f.writelines(line)

    line = '\t\t<bndbox>\n'
    f.writelines(line)

    line = '\t\t\t<xmin>' + xmin + '</xmin>\n'
    f.writelines(line)

    line = '\t\t\t<ymin>' + ymin + '</ymin>\n'
    f.writelines(line)

    line = '\t\t\t<xmax>' + xmax + '</xmax>\n'
    f.writelines(line)

    line = '\t\t\t<ymax>' + ymax + '</ymax>\n'
    f.writelines(line)

    line = '\t\t</bndbox>\n'
    f.writelines(line)

    line = '\t</object>\n'
    f.writelines(line)


def write_xml(xml_fname, corner, bboxes, labels, truncated, image_size=(300, 300, 1)):
    '''
    Write xml file for single image
    :param xml_fname: /path/to/xml_filename
    :param bboxes: np.array (x,4) containing [xmin, ymin, width, height] of each bounding box
    :param labels: list (x,) contating x strings of class names
    :param image_size: size of the image (width, height, depth)
    :return: None
    '''
    # get file name and path information
    base_name = os.path.basename(xml_fname)
    fname = os.path.splitext(base_name)[0]
    parent_folder = os.path.dirname(xml_fname)
    parent_parent_folder = os.path.dirname(parent_folder)

    f = open(xml_fname, 'w')
    width, height, depth = image_size

    line = '<annotation>\n'
    f.writelines(line)

    line = '\t<folder>img</folder>\n'
    f.writelines(line)

    line = '\t<filename>' + fname + '.jpeg' + '</filename>\n'
    f.writelines(line)

    line = '\t<path>' + parent_parent_folder + '</path>\n'
    f.writelines(line)

    line = '\t<source>\n'
    f.writelines(line)

    line = '\t\t<database>50_plex</database>\n'
    f.writelines(line)

    line = '\t\t<corner>' + ",".join(map(str, corner)) + '</corner>\n'
    f.writelines(line)

    line = '\t</source>\n'
    f.writelines(line)

    line = '\t<size>\n'
    f.writelines(line)

    line = '\t\t<width>' + str(width) +'</width>\n'
    f.writelines(line)

    line = '\t\t<height>' + str(height) +'</height>\n'
    f.writelines(line)

    line = '\t\t<depth>3</depth>\n'
    f.writelines(line)

    line = '\t</size>\n'
    f.writelines(line)

    line = '\t<segmented>0</segmented>\n'
    f.writelines(line)

    # write the object information
    for bbox, label, trunc in zip(bboxes, labels, truncated):
        [xmin, ymin, xmax, ymax] = bbox
        write_object(f, str(xmin), str(ymin), str(xmax), str(ymax), str(label), str(trunc))

    line = '</annotation>\n'
    f.writelines(line)

    f.close()


def xml_to_csv(path):
    '''
    Create cvs file for generating cvs file
    :param path: path/to/the/xmls
    :return: path to the saved file name

    Note: keep images in "imgs" folder and xml files in "xmls" folder in same folder
    '''
    xml_list = []
    for xml_file in os.listdir(path):
        tree = ET.parse(os.path.join(path, xml_file))
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    save_path = os.path.dirname(path)
    xml_df.to_csv(os.path.join(save_path, 'labels.csv'), index=None)

    return os.path.join(save_path, 'labels.csv')

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
