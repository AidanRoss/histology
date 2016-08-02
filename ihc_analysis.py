""" This Script identifies useful nest properties for IHC images
Aidan Ross - TDI
@author: Aidan Ross
"""
import csv
import os
import errno


import glob
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from skimage import io
import scipy

# Import useful image analysis modules
from skimage.exposure import rescale_intensity
from skimage.color import rgb2hed, rgb2grey, gray2rgb
from skimage.util import img_as_float, img_as_uint
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.measure import regionprops, ransac, CircleModel
from skimage.morphology import label, remove_small_objects, remove_small_holes, watershed
from skimage.restoration import denoise_nl_means
from skimage.filters import gaussian, threshold_otsu
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter

from scipy import ndimage as ndi

from operator import truediv
from math import pi



def color_conversion(img):
    # This function converts rgb image to the IHC color space, where channel 1 is Hematoxylin, 2 in Eosin and 3 is DAB

    ihc_rgb = skimage.io.imread(img)
    ihc_hed = rgb2hed(ihc_rgb)

    return ihc_rgb, ihc_hed


def rescale(img):
    # Rescaling the Image to a unsigned integer for Otsu thresholding method
    original_img, mask, eos, rr, cc  = segment(img)
    rescaled = rescale_intensity(mask, out_range=(0, 1))
    rescale_masked = rescale_intensity(eos, out_range=(0, 1))

    int_img = img_as_uint(rescaled)
    int_mask = img_as_uint(rescale_masked)

    return int_img, int_mask


def create_bin(img):  # otsu_method=True):
    # Binary image created from Threshold, then labelling is done on this image
    # if otsu_method:
    int_img, masked_int = rescale(img)
    t_otsu = threshold_otsu(int_img)
    bin_img = (int_img >= t_otsu)
    bin_masked = (masked_int >= t_otsu)
    float_img = img_as_float(bin_img)
    float_masked = img_as_float(bin_masked)

    return float_img, float_masked

    # else:
    #    thresh = 400
    #    int_img = rescale(img)
    #    bin_img = (int_img >= thresh)
    #    float_img = img_as_float(bin_img)
    #    return float_img


def segment(img):
    # Identifiying the Tissue punches in order to Crop the image correctly
    im, IHC = color_conversion(img)
    gray = rgb2grey(im)
    smooth = gaussian(gray, sigma=3)
    thresh = 0.88
    binar = (smooth <= thresh)
    bin = remove_small_holes(binar, min_size=200000, connectivity=2)
    bin1 = remove_small_objects(bin, min_size=40000, connectivity=2)
    bin2 = gaussian(bin1, sigma=3)

    eosin = IHC[:, :, 2]
    edges = canny(bin2)
    coords = np.column_stack(np.nonzero(edges))

    model, inliers = ransac(coords, CircleModel, min_samples=3, residual_threshold=1, max_trials=1000)

    rr, cc = circle_perimeter(int(model.params[0]),
                              int(model.params[1]),
                              int(model.params[2]),
                              shape=im.shape)
    a, b = model.params[0], model.params[1]
    r = model.params[2]
    ny, nx = eosin.shape
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    distance = np.sqrt((ix - b)**2 + (iy - a)**2)

    mask = np.ma.masked_where(distance > r, eosin)

    return im, mask, eosin, rr, cc


def label_img(img):
    # Labelling the nests is done using connected components
    img, masked_img = create_bin(img)
    min_nest_size = 100  # Size in Pixels of minimum nest
    min_hole_size = 500  # Size in Pixels of minimum hole

    labeled_img = label(input=masked_img, connectivity=2, background=0)
    rem_holes = remove_small_holes(labeled_img, min_size=min_hole_size, connectivity=2)
    labeled_img1 = remove_small_objects(rem_holes, min_size=min_nest_size, connectivity=2)
    labeled = label(labeled_img1, connectivity=2, background=0)

    print labeled
    return labeled


def display_image(img):
    # Displaying images if needed
    original, ihc_images = color_conversion(img)
    #bin_images, bin_masked = create_bin(img)
    im, mask, eos, rr, cc = segment(img)
    labeled_img = label_img(img)
    n = len(np.unique(labeled_img)) - 1

    plt.figure()
    plt.subplot(141)
    plt.imshow(ihc_images[:, :, 2], cmap=plt.cm.gray)
    plt.title("DAB color space")

    plt.subplot(142)
    plt.imshow(labeled_img, cmap=plt.cm.spectral)
    plt.title("Labeled Image %d" % n)

    plt.subplot(143)
    plt.imshow(mark_boundaries(original, label_img=labeled_img, color=(1, 0, 0)))
    plt.title('Overlay Outlines')

    plt.subplot(144)
    plt.imshow(mask, cmap='gray')
    plt.title('Full Punch Segmentation')


    #return plt.show()


def get_data(img):
    # Obtaining the data for each nest
    labels = label_img(img)
    props = regionprops(labels)

    area = []
    perimeter = []
    eccentricity = []
    filled_area = []
    maj_axis = []
    min_axis = []

    ns = len(np.unique(labels)) - 1
    print ns, 'Number of Nests'

    for seg in range(ns):

        area.append(props[seg].area)
        perimeter.append(props[seg].perimeter)
        eccentricity.append(props[seg].eccentricity)
        filled_area.append(props[seg].filled_area)
        min_axis.append(props[seg].minor_axis_length)
        maj_axis.append(props[seg].major_axis_length)

    avg_area = np.mean(area)
    avg_perimeter = np.mean(perimeter)
    avg_eccentricity = np.mean(eccentricity)
    avg_filled_area = np.mean(filled_area)
    roundness = map(truediv, min_axis, maj_axis)
    new_list = [4 * pi * a for a in area]
    circularity = map(truediv, new_list, (np.square(perimeter)))
    avg_roundness = np.mean(roundness)
    avg_circularity = np.mean(circularity)

    total_nest_area = sum(area)
    total_nest_perim = sum(perimeter)
    std_dev_area = np.std(area)
    std_dev_perimeter = np.std(perimeter)
    std_dev_eccentricity = np.std(eccentricity)
    std_dev_filled_area = np.std(filled_area)
    std_dev_roundness = np.std(roundness)
    std_dev_circularity = np.std(circularity)
    name = os.path.basename(os.path.normpath(img))

    return ns, area, perimeter, eccentricity, filled_area, avg_area, avg_perimeter, avg_eccentricity, avg_filled_area,\
        roundness, circularity, avg_roundness, avg_circularity, total_nest_area, total_nest_perim,\
        std_dev_area, std_dev_perimeter, std_dev_eccentricity, std_dev_filled_area, std_dev_roundness,\
        std_dev_circularity, name


def write_csv(output_data, save_path):
    # Writing the data file to as CSV
    save_out = save_path + '/output_data.csv'

    with open(save_out, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(output_data)


def save_image(save_path, img):  # overlay=True, binary=False, DAB=False):
    # If needed the images can be saved for further analysis of examination
    original_img, ihc_img = color_conversion(img)
    l_img = label_img(img)
    orig = os.path.basename(os.path.normpath(img))
    img_file = mark_boundaries(original_img, label_img=l_img, color=(1, 0, 0))
    # img_file = l_img
    filename = save_path + 'Segments' + orig  # 'Labelled_%s' % + ".png"  # '%s' % img
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write(filename)
    plt.imsave(fname=filename, arr=img_file)


def main():
    # Main function that executes the functions desired above

    # Test data of 3 images - will be much larger data set in the Future
    png_hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/tma-extracted/tma_extracted_png'
    # hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_6/histology/tiff'
    # hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/tma_test'
    # hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_10/tma_extracted_tiff'
    test_hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/test'  # Path with image files (png)
    path = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/save_images/'
    img_set = test_hist
    img_files = glob.glob(img_set + '/*.png')

    output_name = []
    output_nest = []
    output_area = []
    output_perimeter = []
    output_eccentricity = []
    output_filled_area = []
    output_roundness = []
    output_circularity = []

    out_avg_area = []
    out_avg_perim = []
    out_avg_eccen = []
    out_avg_filled = []
    out_avg_roundness = []
    out_avg_circularity = []

    out_tot_area = []
    out_tot_perim = []
    std_dev_area = []
    std_dev_perimeter = []
    std_dev_eccentricity = []
    std_dev_filled_area = []
    std_dev_roundness = []
    std_dev_circularity = []

    for im in img_files:
        display_image(im)
        # save_image(save_path=path, img=im)
        nest, area, perimeter, eccentricity, filled_area, avg_area, avg_perim, avg_eccen, avg_filled, roundness,\
        circularity, avg_roundness, avg_circularity, tot_area, tot_perim, std_area, std_perimeter, std_eccentricity,\
        std_filled_area, std_roundness, std_circularity, name = get_data(im)

        output_name.append(name)
        output_nest.append(nest)
        output_area.append(area)
        output_perimeter.append(perimeter)
        output_eccentricity.append(eccentricity)
        output_filled_area.append(filled_area)
        output_roundness.append(roundness)
        output_circularity.append(circularity)
        out_avg_area.append(avg_area)
        out_avg_perim.append(avg_perim)
        out_avg_eccen.append(avg_eccen)
        out_avg_filled.append(avg_filled)
        out_avg_roundness.append(avg_roundness)
        out_avg_circularity.append(avg_circularity)

        out_tot_area.append(tot_area)
        out_tot_perim.append(tot_perim)

        std_dev_area.append(std_area)
        std_dev_perimeter.append(std_perimeter)
        std_dev_eccentricity.append(std_eccentricity)
        std_dev_filled_area.append(std_filled_area)
        std_dev_roundness.append(std_roundness)
        std_dev_circularity.append(std_circularity)

    output_data = [output_name,
                   output_nest,
                   output_area,
                   std_dev_area,
                   out_tot_area,
                   out_avg_area,
                   output_perimeter,
                   std_dev_perimeter,
                   out_tot_perim,
                   out_avg_perim,
                   output_circularity,
                   std_dev_circularity,
                   out_avg_circularity,
                   output_roundness,
                   std_dev_roundness,
                   out_avg_roundness,
                   output_eccentricity,
                   std_dev_eccentricity,
                   out_avg_eccen,
                   output_filled_area,
                   std_dev_filled_area,
                   out_avg_filled]

    print output_data

    #write_csv(output_data, save_path='/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9')

    plt.show()

if __name__ == "__main__":
    main()
