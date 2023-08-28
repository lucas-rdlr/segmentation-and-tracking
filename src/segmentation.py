import numpy as np
import torch

import tifffile as tiff
import os

from skimage import filters, measure, morphology, util
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from scipy import ndimage as ndi


def make_mask(img, img_limit=0.05):
    if torch.is_tensor(img):
        img = img.numpy()

    mask = img > img_limit
    mask_labeled = np.vectorize(label, signature='(n,m)->(n,m)')(mask)
    mask = mask_labeled > 0
    mask_fill = np.vectorize(ndi.binary_fill_holes, signature='(n,m)->(n,m)')(mask)
    mask_last = np.vectorize(remove_pixels, signature='(n,m)->(n,m)')(mask_fill)

    return mask_last


#####################################################################


def remove_pixels(slc, area_limit=50):
    new_mask = slc.copy()
    labels = label(slc, connectivity=1, background=0)
    rps = regionprops(labels)
    areas = np.array([r.area for r in rps])
    idxs = np.where(areas<area_limit)[0]

    for i in idxs:
        new_mask[tuple(rps[i].coords.T)] = 0

    return new_mask


#####################################################################


def folder_mask(cell_type, time):

    folder = f'content/Segmentation/training_data/{cell_type}/{time}_MK'
    if os.path.isdir(folder):
        print(f'{cell_type} at time {time} already has a mask')
        return
    
    os.makedirs(folder)
    path = f'content/Segmentation/training_data/{cell_type}/{time}'
    imgs = os.listdir(path)

    for name in imgs:
        file = f'{path}/{name}'
        array = tiff.imread(file) / 255
        mask = make_mask(array)
        tiff.imwrite(f'{folder}/{name}', mask)
    
    return


#####################################################################


def post_processing(image, radius=4, thres=10, binary=True):
    """
    Function to post-process the masks output by the Neural Networks so that the
    segmented regions are filled and noise is left out.

    Inputs:
        - image: batch image output by the NN with shape (B, H, W).
        - radius: radious of disk regions to use as filter to compute the mean.
        - thres: value to left noise out after computing the mean.
        - binary: boolean indicating wether to perform semantic segmentation or
                  instance (multiple classes).

    Output: array of shape (B, H, W) containing the new masks.
    """

    if type(image) is not np.ndarray:
        image = np.array(image)

    new_image = np.zeros_like(image, dtype=int)

    for i,img in enumerate(image):
        threshold = filters.threshold_otsu(img)

        cells = img > threshold

        smoother_dividing = filters.rank.mean(util.img_as_ubyte(cells),
                                            morphology.disk(radius))

        binary_smoother_dividing = smoother_dividing > thres

        fill = ndi.binary_fill_holes(binary_smoother_dividing)
        cleaned_dividing = measure.label(fill)
        cleaned = remove_pixels(cleaned_dividing, 1000)

        if binary:
            cleaned = cleaned > 0

        new_image[i,:,:] = cleaned

    return new_image