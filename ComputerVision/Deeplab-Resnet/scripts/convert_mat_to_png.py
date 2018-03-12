import os
import argparse

import scipy.io
import skimage.io as io


def convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_augmented_root):
    """ Creates a new folder in the root folder of the dataset with annotations stored in .png.
    The function accepts a full path to the root of Berkeley augmented Pascal VOC segmentation
    dataset and converts annotations that are stored in .mat files to .png files. It creates
    a new folder dataset/cls_png where all the converted files will be located. If this
    directory already exists the function does nothing. The Berkley augmented dataset
    can be downloaded from here:
    http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

    Parameters
    ----------
    pascal_berkeley_augmented_root : string
        Full path to the root of augmented Berkley PASCAL VOC dataset.
    """
    def read_class_annotation_array_from_berkeley_mat(mat_filename, key='GTcls'):
        #  Mat to png conversion for http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html
        # 'GTcls' key is for class segmentation
        # 'GTinst' key is for instance segmentation
        #  Credit: https://github.com/martinkersner/train-DeepLab/blob/master/utils.py
        mat = scipy.io.loadmat(mat_filename, mat_dtype=True, squeeze_me=True, struct_as_record=False)
        return mat[key].Segmentation

    mat_file_extension_string = '.mat'
    png_file_extension_string = '.png'
    relative_path_to_annotation_mat_files = 'dataset/cls'
    relative_path_to_annotation_png_files = 'dataset/cls_png'

    annotation_mat_files_fullpath = os.path.join(pascal_berkeley_augmented_root,
                                                 relative_path_to_annotation_mat_files)
    annotation_png_save_fullpath = os.path.join(pascal_berkeley_augmented_root,
                                                relative_path_to_annotation_png_files)
    # Create the folder where all the converted png files will be placed
    # If the folder already exists, do nothing
    if not os.path.exists(annotation_png_save_fullpath):
        os.makedirs(annotation_png_save_fullpath)
    else:
        return

    for current_mat_file_name in os.listdir(annotation_mat_files_fullpath):
        current_file_name_without_extention = current_mat_file_name[:-len(mat_file_extension_string)]
        current_mat_file_full_path = os.path.join(annotation_mat_files_fullpath, current_mat_file_name)
        current_png_file_full_path_to_be_saved = os.path.join(annotation_png_save_fullpath, current_file_name_without_extention)
        current_png_file_full_path_to_be_saved += png_file_extension_string

        annotation_array = read_class_annotation_array_from_berkeley_mat(current_mat_file_full_path)

        io.imsave(current_png_file_full_path_to_be_saved, annotation_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SBD mat to png')
    parser.add_argument('--sbd_path', type=str, required=True, help='SBD PATH')
    args = parser.parse_args()

    convert_pascal_berkeley_augmented_mat_annotations_to_png(args.sbd_path)
