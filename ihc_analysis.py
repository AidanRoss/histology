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
from skimage.color import rgb2hed, rgb2grey
from skimage.util import img_as_float, img_as_uint
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.measure import regionprops, ransac, CircleModel
from skimage.morphology import label, remove_small_objects, remove_small_holes
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
    # original_img, mask, dab, rr, cc = segment(img)
    # b = mask.data
    # rescaled_mask = rescale_intensity(b, out_range=(0, 1))
    orig, ihc = color_conversion(img)
    rescaled = rescale_intensity(ihc[:, :, 2], out_range=(0, 1))
    int_img = img_as_uint(rescaled)

    #int_mask_data = img_as_uint(rescaled_mask)

    return int_img, orig, ihc  # , int_mask_data


def create_bin(img):  # otsu_method=True):
    # Binary image created from Threshold, then labelling is done on this image
    # if otsu_method:
    int_img, orig, ihc = rescale(img)
    t_otsu = threshold_otsu(int_img)
    bin_img = (int_img >= t_otsu)
    # bin =

    float_img = img_as_float(bin_img)

    # float_masked = img_as_float(bin_masked)

    return float_img, orig, ihc  # , float_masked


def segment(img):
    # Identifiying the Tissue punches in order to Crop the image correctly
    # Canny edges and RANSAC is used to fit a circe to the punch
    # A Mask is created

    distance = 0
    r = 0

    float_im, orig, ihc = create_bin(img)
    gray = rgb2grey(orig)
    smooth = gaussian(gray, sigma=1)
    thresh = 0.88
    binar = (smooth <= thresh)

    bin = remove_small_holes(binar, min_size=200000, connectivity=2)
    bin1 = remove_small_objects(bin, min_size=40000, connectivity=2)
    bin2 = gaussian(bin1, sigma=1)
    bin3 = (bin2 > 0)

    # eosin = IHC[:, :, 2]
    edges = canny(bin3)
    coords = np.column_stack(np.nonzero(edges))

    model, inliers = ransac(coords, CircleModel, min_samples=6, residual_threshold=1, max_trials=1000)

    # rr, cc = circle_perimeter(int(model.params[0]),
    #                          int(model.params[1]),
    #                          int(model.params[2]),
    #                          shape=im.shape)
    a, b = model.params[0], model.params[1]
    r = model.params[2]
    ny, nx = bin3.shape
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    distance = np.sqrt((ix - b)**2 + (iy - a)**2)

    mask = np.ma.masked_where(distance > r, bin3)

    return distance, r, float_im, orig, ihc, bin3


def label_img(img):
    # Labelling the nests is done using connected components
    dist = 0
    radius = 0
    dist, radius, float_img, orig, ihc, bin3 = segment(img)         # '''fix'''
    masked_img = np.ma.masked_where(dist > radius, float_img)
    masked_bool = np.ma.filled(masked_img, fill_value=0)

    min_nest_size = 100  # Size in Pixels of minimum nest
    min_hole_size = 500  # Size in Pixels of minimum hole

    labeled_img = label(input=masked_bool, connectivity=2, background=0)
    rem_holes = remove_small_holes(labeled_img, min_size=min_hole_size, connectivity=2)
    labeled_img1 = remove_small_objects(rem_holes, min_size=min_nest_size, connectivity=2)
    labeled = label(labeled_img1, connectivity=2, background=0)
    mask_lab = np.ma.masked_where(dist > radius, labeled)

    print labeled

    return labeled, masked_img, orig, ihc, bin3


def display_image(img):
    # Displaying images if needed

    labeled_img, masked_img, orig, ihc, bin3 = label_img(img)
    n = len(np.unique(labeled_img)) - 1

    plt.figure()
    plt.subplot(141)
    plt.imshow(bin3, cmap='gray')
    plt.title("DAB color space")

    plt.subplot(142)
    plt.imshow(labeled_img, cmap=plt.cm.spectral)
    plt.title("Labeled Image %d" % n)

    plt.subplot(143)
    plt.imshow(mark_boundaries(orig, label_img=labeled_img, color=(1, 0, 0)))
    plt.title('Overlay Outlines')

    plt.subplot(144)
    plt.imshow(masked_img, cmap='gray')
    plt.title('Full Punch Segmentation')


    #return plt.show()


def get_data(img):
    # Obtaining the data for each nest
    labels, mask, orig, ihc, bin3 = label_img(img)
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


    png_hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/tma-extracted/tma_extracted_png'
    test_hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_12/test'  # Path with image files (png)
    path = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/save_images/' # Path to save CSV file

    ### Uncomment this to run - Raquel
    #test_hist = '/Users/engs1348/Raquel/Nottingham-TMAs/tma-extracted'
    #png_hist = '/Users/engs1348/Raquel/Nottingham-TMAs/tma-extracted'
    #path = '/Users/engs1348/Raquel/githubRepositoryWorkingFiles/Histology_Aidan'

    img_set = test_hist  # Image set that is to be analyzed
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
        ### Uncomment below if it is desired to save images and obtain data
        # save_image(save_path=path, img=im)
        #nest, area, perimeter, eccentricity, filled_area, avg_area, avg_perim, avg_eccen, avg_filled, roundness,\
        #circularity, avg_roundness, avg_circularity, tot_area, tot_perim, std_area, std_perimeter, std_eccentricity,\
        #std_filled_area, std_roundness, std_circularity, name = get_data(im)
#
        #output_name.append(name)
        #output_nest.append(nest)
        #output_area.append(area)
        #output_perimeter.append(perimeter)
        #output_eccentricity.append(eccentricity)
        #output_filled_area.append(filled_area)
        #output_roundness.append(roundness)
        #output_circularity.append(circularity)
        #out_avg_area.append(avg_area)
        #out_avg_perim.append(avg_perim)
        #out_avg_eccen.append(avg_eccen)
        #out_avg_filled.append(avg_filled)
        #out_avg_roundness.append(avg_roundness)
        #out_avg_circularity.append(avg_circularity)
#
        #out_tot_area.append(tot_area)
        #out_tot_perim.append(tot_perim)
#
        #std_dev_area.append(std_area)
        #std_dev_perimeter.append(std_perimeter)
        #std_dev_eccentricity.append(std_eccentricity)
        #std_dev_filled_area.append(std_filled_area)
        #std_dev_roundness.append(std_roundness)
        #std_dev_circularity.append(std_circularity)

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

    # print output_data

    #write_csv(output_data, save_path='/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9')

    ##output_path = '/Users/engs1348/Raquel/githubRepositoryWorkingFiles/Histology_Aidan'
    #write_csv(output_data, save_path=output_path)

    plt.show()

if __name__ == "__main__":
    main()
