import os
import glob

SET = '512'
with open('../TASK_NAME') as f:
    TASK = f.readline().strip()
PATH = '../dataset'
SAVE = '../train_test_list/{}'.format(TASK)

if not os.path.isdir(SAVE):
    os.makedirs(SAVE)

def save(img_groups, prefix, folder):
    print('{}/{}: {}'.format(folder, prefix, len(img_groups)))
    with open(os.path.join(folder, '{}_{}.csv'.format(prefix, SET)), 'w') as f:
        f.write('\n'.join([','.join(line) for line in img_groups]))

def generate_list(set_name):
    with open('{}_idx.txt'.format(set_name)) as f:
        idxs = [int(line.strip()) for line in f]

    img_groups = []
    for idx in idxs:
        imgs = glob.glob(os.path.join(PATH, 'rgb', SET, 'a{:04d}-*.tif'.format(idx)))
        for full_path in imgs:
            path = os.path.basename(full_path)
            rgb_path = os.path.join('rgb', SET, path)
            gt_path  = os.path.join(TASK,  SET, path)

            assert os.path.isfile(os.path.join(PATH, rgb_path))
            assert os.path.isfile(os.path.join(PATH, gt_path))

            img_groups.append((rgb_path, gt_path))

    save(img_groups, set_name, SAVE)

generate_list('train')
generate_list('test')