SAVE_PATH = '../../dataset/style_transfer/512';
imgs = dir('../../dataset/rgb/512/*.tif');

%% load images
M = imread('images/ruins.png');
M = rgb2gray(double(M)./255);

mkdir(SAVE_PATH);
parfor idx = 1:length(imgs)
    path = fullfile(imgs(idx).folder, imgs(idx).name);
    im = imread(path);
    im = rgb2gray(double(im)./65535);

    tic
    [gt, ~] = style_transfer(im, M, 10, 4);
    toc

    save_path = fullfile(SAVE_PATH, imgs(idx).name);
    imwrite(gt, save_path);
end
