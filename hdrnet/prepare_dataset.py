import os
from utils import tiff2png

with open('../SU/TASK_NAME') as f:
    TASK = f.readline().strip()
SRC_PATH = '../SU/dataset'
SRC_LIST = '../SU/train_test_list/{}/train_512.csv'.format(TASK)
SAVE_PATH = 'dataset/{}'.format(TASK)
for folder in ['input', 'output']:
    path = os.path.join(SAVE_PATH, folder)
    if not os.path.isdir(path):
        os.makedirs(path)

img_list = []
with open(SRC_LIST) as f:
    imgs = [l.strip().split(',') for l in f]

for idx, (x_path, y_path) in enumerate(imgs):
    assert os.path.basename(x_path) == os.path.basename(y_path)
    assert os.path.isfile(os.path.join(SRC_PATH, x_path))
    assert os.path.isfile(os.path.join(SRC_PATH, y_path))

    name = os.path.basename(x_path).replace('.tif', '.png')
    tiff2png(os.path.abspath(os.path.join(SRC_PATH, x_path)), os.path.join(SAVE_PATH, 'input', name))
    tiff2png(os.path.abspath(os.path.join(SRC_PATH, y_path)), os.path.join(SAVE_PATH, 'output', name))

    img_list.append(name)

    print('Processing {}/{} ...'.format(idx, len(imgs)))

with open(os.path.join(SAVE_PATH, 'filelist.txt'), 'w') as f:
    f.write('\n'.join(img_list))