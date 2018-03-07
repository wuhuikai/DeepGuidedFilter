from skimage.io import imread, imsave

def tiff2png(in_path, out_path):
    assert '.png' in out_path
    imsave(out_path, imread(in_path))