SAVE_PATH = '../../dataset/l0_smooth/512';
imgs = dir('../../dataset/rgb/512/*.tif');

mkdir(SAVE_PATH);
parfor idx = 1:length(imgs)
    path = fullfile(imgs(idx).folder, imgs(idx).name);
    im = imread(path);
    gt = L0Smoothing(im, 0.01);
    
    save_path = fullfile(SAVE_PATH, imgs(idx).name);
    imwrite(gt, save_path);
end