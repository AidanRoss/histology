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

# Import useful image analysis modules
from skimage.exposure import rescale_intensity
from skimage.color import rgb2hed
from skimage.util import img_as_float, img_as_uint
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.measure import regionprops
from skimage.morphology import label, remove_small_objects, remove_small_holes
from mahotas import otsu


def color_conversion(img_files):

    ihc_rgb = skimage.io.imread(im)
    ihc_hed = rgb2hed(ihc_rgb)
    
    return ihc_rgb, ihc_hed

def rescale(img):

    original_img, rescale_img = color_conversion(img)
    rescale = rescale_intensity(rescale_img[:, :, 2], out_range=(0, 1))
    int_img = img_as_uint(rescale)

    return int_img


def create_bin(img):

    int_img = rescale(img)
    t_otsu = otsu(int_img)
    bin = (int_img >= t_otsu)
    float_img = img_as_float(bin)

    return float_img


def segment(img):

    img = create_bin(img)
    fz_seg = felzenszwalb(img, scale=100, sigma=0.51, min_size=200)
    
    print fz_seg
    return fz_seg


def label_img(img_files):

    img = create_bin(img_files)
    labeled_img = label(input=img, connectivity=2, background=0)
    labeled_img = remove_small_objects(labeled_img, min_size=300, connectivity=2)

    print labeled_img
    return labeled_img


def display_images(img):

    original, ihc_images = color_conversion(img)
    bin_images = create_bin(img_files)
    fz_seg = segment(img_files)
    labeled_img = label_img(img_files)

    plt.figure()

    plt.subplot(131)
    plt.imshow(ihc_images[:, :, 2], cmap=plt.cm.gray)
    plt.title("DAB color space")

    plt.subplot(132)
    plt.imshow(labeled_img, cmap=plt.cm.spectral)
    plt.title("Labeled Image")

    plt.subplot(133)
    plt.imshow(mark_boundaries(original, label_img=labeled_img, color=(1, 0, 0)))
    plt.title('Overlay Outlines')

    #return plt.show()

def get_data(img):

    labels = label_img(img)
    props = regionprops(labels)

    area = []
    perimeter = []
    eccentricity = []
    filled_area = []

    ns = len(np.unique(labels))
    print ns, ' Is the Number of Labelled Regions'

    for seg in range(ns-1):

        area.append(props[seg].area)
        perimeter.append(props[seg].perimeter)
        eccentricity.append(props[seg].eccentricity)
        filled_area.append(props[seg].filled_area)
        
    avg_area = np.mean(area)
    avg_perimeter = np.mean(perimeter)
    avg_eccentricity = np.mean(eccentricity)
    avg_filled_area = np.mean(filled_area)

    return area, perimeter, eccentricity, filled_area, avg_area, avg_perimeter, avg_eccentricity, avg_filled_area


def write_csv(output_data, save_path):

    save_out = save_path + '/output_data.csv'
    with open(save_out, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(output_data)


def save_image(save_path, img_name):

    filename = save_path + '/' + img_name + ".tiff"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write(filename)
    plt.imsave(fname=save_path + '/' + str(i) + ".tiff", arr=mark_boundaries(ihc_rgb, fz_seg, color=(1, 0, 0)))


def main():

    # Test data of 3 images - will be much larger data set in the Future
    # hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_6/histology/tiff'
    hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/tma_test'
    path = 'User/aidan/desktop/aidan_summer/Week_Tasks/Week_7'
    img_set = hist
    img_files = glob.glob(img_set + '/*.tif')
    output_area = []
    output_perimeter = []
    output_eccentricity = []
    output_filled_area = []
    out_avg_area = []
    out_avg_perim = []
    out_avg_eccen = []
    out_avg_filled = []
    
    for im in img_files:
        display_images(im)
        area, perimeter, eccentricity, filled_area, avg_area, avg_perim, avg_eccen, avg_filled = get_data(im)
        
        output_area.append(area)
        output_perimeter.append(perimeter)
        output_eccentricity.append(eccentricity)
        output_filled_area.append(filled_area)
        out_avg_area.append(avg_area)
        out_avg_perim.append(avg_perim)
        out_avg_eccen.append(avg_eccen)
        out_avg_filled.append(avg_filled)
        
    output_data = [output_area,
                    output_perimeter,
                    output_eccentricity,
                    output_filled_area,
                    out_avg_area
                    out_avg_eccen,
                    out_avg_filled]
                    
    print output_data
    
    write_csv(output_data, save_path='/Users/aidan/desktop/aidan_summer/Week_Tasks/Week_9/)
    
    plt.show()

if __name__ == "__main__":
    main()

