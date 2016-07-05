"""The Script segments IHC images and calculates important segment and image characteristics for further analysis
Aidan Ross - TDI"""
# Import basic Python packages
import csv
import sys
import os, errno
import glob
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

# Import image analysis modules
import skimage.filters
from skimage.exposure import rescale_intensity
from skimage.color import rgb2hed
from skimage.util import img_as_float, img_as_uint
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.measure import regionprops
from mahotas import otsu

# Test data of 3 images - will be much larger data set in the Future
hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_6/histology/tiff'
path = 'User/aidan/desktop/aidan_summer/Week_Tasks/Week_7'


def main(img_set, csv=False, display=False, save_path=False, save_im=False):

    img_files = glob.glob(img_set + '/*.tif')
    output_area = []
    output_perimeter = []

    for i, im in enumerate(img_files):

        ihc_rgb = skimage.io.imread(im)
        ihc_hed = rgb2hed(ihc_rgb)

        # Hematoxylin = ihc_hed[:, :, 0] ## Channel 1 displays Hematoxylin
        # Eosin = ihc_hed[:, :, 1]       ## Channel 2 displays Eosin
        # DAB = ihc_hed[:, :, 2]         ## Channel 3 displays DAB

        # Rescaling the intensity to be between 0-1 for only the DAB channel
        rescale = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1))
        int_img = img_as_uint(rescale)

        # Now use Otsu's thresholding method to calculate threshold
        '''Otsu's thresholding method calculates a global threshold minimizes the intra-class variance -
        defined as the weighted sum of variances of the two classes'''
        T_otsu = otsu(int_img)

        # Create a binary image using the threshold
        bin = (int_img >= T_otsu)

        # Felzenszwalb segmentation method accepts images as floats so we need to change the image back to a float
        img1 = img_as_float(bin)

        '''Felzenszwalb segmentation performs a graph based segmentation algorithm - the method is baesd
        on selecting edges from a graph, where each pixel corresponds to a node in  the graph an certain neighbouring
        pixels are connected via undirected edges ( Source: Felzenszwalb Paper) '''

        fz_seg = felzenszwalb(img1, scale=2000, sigma=1.4, min_size=850)

        print fz_seg

        print("Felzenszwalb's number or segments: %d" % len(np.unique(fz_seg)))

        # If we want to display the images we set display to be "True" when we call the function - good for testing
        # of small image sets

        if display:

            # First Figure Displays 5 images : Original, Hematoxylin, Eosin, DAB, Felzenszwalb Segmentation - Spectral
            plt.figure(num=i)
            plt.subplot(151)
            plt.imshow(ihc_rgb)
            plt.title("Original image")

            plt.subplot(152)
            plt.imshow(ihc_hed[:, :, 0], cmap=plt.cm.gray)
            plt.title("Hematoxylin")

            plt.subplot(153)
            plt.imshow(ihc_hed[:, :, 1], cmap=plt.cm.gray)
            plt.title("Eosin")

            plt.subplot(154)
            plt.imshow(ihc_hed[:, :, 2], cmap=plt.cm.gray)
            plt.title("DAB")

            plt.subplot(155)
            plt.imshow(fz_seg, cmap=plt.cm.spectral)
            plt.title("Felzenszwalb Segmentation - Spectral")

            # Second Figure Displays three images - Gray segmentation, segments and binary image
            plt.figure(num=i+len(img_files))
            plt.subplot(131)
            plt.imshow(fz_seg, cmap=plt.cm.gray)
            plt.title("Felzenszwalb's Segmentation - Gray")

            plt.subplot(132)
            plt.imshow(mark_boundaries(ihc_rgb, fz_seg, color=(1,0,0)))
            plt.title('Segments')

            plt.subplot(133)
            plt.imshow(bin, cmap=plt.cm.gray)
            plt.title("Binary Image that is segmented")

        # Calculate the useful physical properties of the segmented regions using regionprops function from scikit image
        props = regionprops(fz_seg)

        # ns = number of segments
        ns = len(np.unique(fz_seg))
        print("Felzenszwalb's number or segments: %d" % ns)

        # Initialize the parameters to be an empty list
        #centroid = []
        area = []
        perimeter = []

        # For each image : calculate the centroid, area, and perimeter for each segment - possibly average after?
        for seg in range(ns-1):
            # centroid.append(props[seg].centroid)
            area.append(props[seg].area)
            perimeter.append(props[seg].perimeter)

        output_area.append(area)
        output_perimeter.append(perimeter)
        # explore other regionprops functions or functions outside of regionprops that might be useful
        # maybe the average protein area and perimeter per image would be good 
        print output_area
        print output_perimeter
        lst = [output_area,
               output_perimeter]
        if csv:
            save_out = save_path + '/output_data.csv'
            with open(save_out, "wb") as f:
                writer = csv.writer(f)
                writer.writerows(lst)
    plt.show()

        #if save_im:
        #    filename = save_path +'/' + str(i) + ".tiff"
        #    if not os.path.exists(os.path.dirname(filename)):
        #        try:
        #            os.makedirs(os.path.dirname(filename))
        #        except OSError as exc:  # Guard against race condition
        #            if exc.errno != errno.EEXIST:
        #                raise
        #    with open(filename, "w") as f:
        #        f.write(filename)
        #    plt.imsave(fname=save_path +'/' + str(i) + ".tiff", arr=mark_boundaries(ihc_rgb, fz_seg, color=(1,0,0)))
    #plt.show()

if __name__ == "__main__":
    main(img_set=hist, display=True)

