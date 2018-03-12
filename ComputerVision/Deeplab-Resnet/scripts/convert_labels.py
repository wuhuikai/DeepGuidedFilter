#!/usr/bin/env python
# Martin Kersner, m.kersner@gmail.com
# 2016/01/25

from __future__ import print_function

import os
import sys

import numpy as np

from skimage.io import imread, imsave


def pascal_palette():
    palette = {(0, 0, 0): 0,
               (128, 0, 0): 1,
               (0, 128, 0): 2,
               (128, 128, 0): 3,
               (0, 0, 128): 4,
               (128, 0, 128): 5,
               (0, 128, 128): 6,
               (128, 128, 128): 7,
               (64, 0, 0): 8,
               (192, 0, 0): 9,
               (64, 128, 0): 10,
               (192, 128, 0): 11,
               (64, 0, 128): 12,
               (192, 0, 128): 13,
               (64, 128, 128): 14,
               (192, 128, 128): 15,
               (0, 64, 0): 16,
               (128, 64, 0): 17,
               (0, 192, 0): 18,
               (128, 192, 0): 19,
               (0, 64, 128): 20}

    return palette


def convert_from_color_segmentation(arr_3d):
    arr_2d = np.ones((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8) * 255
    palette = pascal_palette()

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def main():
    path, txt_file, path_converted = process_arguments(sys.argv)

    # Create dir for converted labels
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    with open(txt_file, 'r') as f:
        for img_name in f:
            img_base_name = img_name.strip()
            img_name = os.path.join(path, img_base_name) + '.png'
            img = imread(img_name)

            if (len(img.shape) > 2):
                img = convert_from_color_segmentation(img)
                imsave(os.path.join(path_converted, img_base_name) + '.png', img)
            else:
                print(img_name,  "is not composed of three dimensions, therefore "
                                 "shouldn't be processed by this script.\n"
                                 "Exiting.", file=sys.stderr)
                exit()


def process_arguments(argv):
    if len(argv) != 4:
        help()

    path = argv[1]
    list_file = argv[2]
    new_path = argv[3]

    return path, list_file, new_path


def help():
    print('Usage: python convert_labels.py PATH LIST_FILE NEW_PATH\n'
          'PATH points to directory with segmentation image labels.\n'
          'LIST_FILE denotes text file containing names of images in PATH.\n'
          'Names do not include extension of images.\n'
          'NEW_PATH points to directory where converted labels will be stored.'
          , file=sys.stderr)
    exit()

if __name__ == '__main__':
    main()
