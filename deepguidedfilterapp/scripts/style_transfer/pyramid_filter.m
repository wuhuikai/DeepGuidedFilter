% This is a 2D separable low pass filter for constructing Gaussian and 
% Laplacian pyramids, built from a 1D 5-tap low pass filter.
%
% tom.mertens@gmail.com, August 2007
% sam.hasinoff@gmail.com, March 2011  [imfilter faster with 2D filter]
%

function f = pyramid_filter()
f = [.05, .25, .4, .25, .05];  % original [Burt and Adelson, 1983]
%f = [.0625, .25, .375, .25, .0625];  % binom-5
f = f'*f;
end