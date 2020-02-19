import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import PIL.ImageDraw as ImageDraw
from skimage.io import imsave

def imadjust(image, tol=[0.01, 0.99]):
    # img : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 1.

    def adjust_single_channel(img, tol):
        if img.dtype == 'uint8':
            nbins = 255
        elif img.dtype == 'uint16':
            nbins = 65535

        N = np.histogram(img, bins=nbins, range=[0, nbins])  # get histogram of image
        cdf = np.cumsum(N[0]) / np.sum(N[0])  # calculate cdf of image
        ilow = np.argmax(cdf > tol[0]) / nbins  # get lowest value of cdf (normalized)
        ihigh = np.argmax(cdf >= tol[1]) / nbins  # get heights value of cdf (normalized)

        lut = np.linspace(0, 1, num=nbins)  # create convert map of values
        lut[lut <= ilow] = ilow  # make sure they are larger than lowest value
        lut[lut >= ihigh] = ihigh  # make sure they are smaller than largest value
        lut = (lut - ilow) / (ihigh - ilow)  # normalize between 0 and 1
        lut = np.round(lut * nbins).astype(img.dtype)  # convert to the original image's type

        img_out = np.array(
            [[lut[i] for i in row] for row in img])  # convert input image values based on conversion list

        return img_out



def visualize_bbxs(image, centers=None, bbxs=None, save=False, adjust_hist=False, dpi=80):
    # adjust the histogram of the image
    if adjust_hist:
        image = imadjust(image)

    fig_height, fig_width = image.shape[:2]
    figsize = fig_width / float(dpi), fig_height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(image)
    if centers is not None:
        ax.plot(centers[:, 0], centers[:, 1], 'b.')
    if bbxs is not None:
        for idx in range(bbxs.shape[0]):
            xmin = bbxs[idx, 0]
            ymin = bbxs[idx, 1]
            width = bbxs[idx, 2] - bbxs[idx, 0]
            height = bbxs[idx, 3] - bbxs[idx, 1]
            ax.add_patch(patches.Rectangle((xmin, ymin), width, height,
                                           edgecolor="blue", fill=False))
    if save:
        fig.savefig('visualized_image.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def bbxs_image(file_name, bbxs, image_size, color=None):
    '''
    Save RGB image with bounding boxes
    :param file_name: tifffile to be saved
    :param bbxs: np.array [xmin ymin xmax ymax]
    :param image_size: [width height]
    :param color: color of bounding box 'red', 'blue', None: gray image
    :return:
    '''

    if color is None:
        box_pil = Image.new('L', image_size)
        color = 'white'
    else:
        box_pil = Image.new('RGB', image_size)
    box_draw = ImageDraw.Draw(box_pil)

    for xmin, ymin, xmax, ymax in bbxs:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        box_draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=2, fill=color)
    try:
        box_pil.save(file_name)
    except:
        imsave(file_name, np.array(box_pil), plugin='tifffile', bigtiff=True)


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
