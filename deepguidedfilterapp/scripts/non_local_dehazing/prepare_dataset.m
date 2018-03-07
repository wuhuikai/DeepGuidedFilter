SAVE_PATH = '../../dataset/non_local_dehazing/512';
imgs = dir('../../dataset/rgb/512/*.tif');

mkdir(SAVE_PATH);
parfor idx = 1:length(imgs)
    path = fullfile(imgs(idx).folder, imgs(idx).name);
    im = imread(path);

    gamma = 1;
    A = reshape(estimate_airlight(im2double(im).^(gamma)),1,1,3);
    [gt, ~] = non_local_dehazing(im, A, gamma);

    save_path = fullfile(SAVE_PATH, imgs(idx).name);
    imwrite(gt, save_path);
end