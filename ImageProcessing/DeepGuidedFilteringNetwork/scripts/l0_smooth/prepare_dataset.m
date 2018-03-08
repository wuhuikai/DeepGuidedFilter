function prepare_dataset(image_size)
    SAVE_PATH = strcat('../../dataset/l0_smooth/', image_size);
    imgs = dir(strcat('../../dataset/rgb/', image_size, '/*.tif'));

    mkdir(SAVE_PATH);
    parfor idx = 1:length(imgs)
        path = fullfile(imgs(idx).folder, imgs(idx).name);
        im = imread(path);
        gt = L0Smoothing(im, 0.01);

        save_path = fullfile(SAVE_PATH, imgs(idx).name);
        imwrite(gt, save_path);
    end
end