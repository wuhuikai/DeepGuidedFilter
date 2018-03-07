% Contruction of Laplacian pyramid
%
% Arguments:
%   image 'I'
%   'nlev', number of levels in the pyramid (optional)
%   subwindow indices 'subwindow', given as [r1 r2 c1 c2] (optional) 
%
% tom.mertens@gmail.com, August 2007
% sam.hasinoff@gmail.com, March 2011  [modified to handle subwindows]
%
%
% More information:
%   'The Laplacian Pyramid as a Compact Image Code'
%   Burt, P., and Adelson, E. H., 
%   IEEE Transactions on Communication, COM-31:532-540 (1983). 
%

function pyr = laplacian_pyramid(I,nlev,subwindow)

r = size(I,1);
c = size(I,2);
if ~exist('subwindow','var')
    subwindow = [1 r 1 c];
end
if ~exist('nlev','var')
    nlev = numlevels([r c]);  % build highest possible pyramid
end

% recursively build pyramid
pyr = cell(nlev,1);
filter = pyramid_filter;
J = I;
for l = 1:nlev - 1
    % apply low pass filter, and downsample
    [I,subwindow_child] = downsample(J,filter,subwindow);
    
    % in each level, store difference between image and upsampled low pass version
    pyr{l} = J - upsample(I,filter,subwindow);

    J = I; % continue with low pass image
    subwindow = subwindow_child;
end
pyr{nlev} = J; % the coarest level contains the residual low pass image

  


