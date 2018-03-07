% Construction of Gaussian pyramid
%
% Arguments:
%   image 'I'
%   'nlev', number of levels in the pyramid (optional)
%   subwindow indices 'subwindow', given as [r1 r2 c1 c2] (optional) 
%
% tom.mertens@gmail.com, August 2007
% sam.hasinoff@gmail.com, March 2011  [modified to handle subwindows]
%

function pyr = gaussian_pyramid(I,nlev,subwindow)

r = size(I,1);
c = size(I,2);
if ~exist('subwindow','var')
    subwindow = [1 r 1 c];
end
if ~exist('nlev','var')    
    nlev = numlevels([r c]);  % build highest possible pyramid
end

% start by copying the image to the finest level
pyr = cell(nlev,1);
pyr{1} = I;

% recursively downsample the image
filter = pyramid_filter;
for l = 2:nlev
    I = downsample(I,filter,subwindow);
    pyr{l} = I;
end