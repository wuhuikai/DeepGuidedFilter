SAVE_PATH = '../../dataset/multi_scale_detail_manipulation/512';
imgs = dir('../../dataset/rgb/512/*.tif');

mkdir(SAVE_PATH);
parfor idx = 1:length(imgs)
    path = fullfile(imgs(idx).folder, imgs(idx).name);
    im = imread(path);
    
    gt = multi_scale_detail_manipulation(im);
    
    save_path = fullfile(SAVE_PATH, imgs(idx).name);
    imwrite(gt, save_path);
end