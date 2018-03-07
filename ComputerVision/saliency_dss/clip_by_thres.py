import argparse
import os
import glob
import tqdm

import numpy as np

from skimage.io import imread, imsave

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='Data root')
    parser.add_argument('thres', type=int, help='Threshold')
    args = parser.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.root, '*_sal.png')), key=lambda s: int(os.path.basename(s).split('_')[0]))

    save = []
    for im_path in tqdm.tqdm(imgs, total=len(imgs), ncols=80, leave=False):
        pre_im = imread(im_path)
        pre_im[pre_im >  args.thres] = 255
        pre_im[pre_im <= args.thres] =   0

        gt_im = imread(im_path.replace('_sal', ''))
        save.append(os.path.basename(im_path)+','+str(np.mean(np.abs(pre_im - gt_im))))

        imsave(im_path.replace('sal', 'sal_{}'.format(args.thres)), pre_im)

    with open(args.root+'_thres_mae.csv', 'w') as f:
        f.write('\n'.join(save))

if __name__ == '__main__':
    main()