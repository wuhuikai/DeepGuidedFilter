function out = wls_optimization(in, data_weight, guidance, lambda)
%Weighted Least Squares optimization solver.
% Given an input image IN, we seek a new image OUT, which, on the one hand,
% is as close as possible to IN, and, at the same time, is as smooth as
% possible everywhere, except across significant gradients in the hazy image.
%
%  Input arguments:
%  ----------------
%  in             - Input image (2-D, double, N-by-M matrix).   
%  data_weight    - High values indicate it is accurate, small values
%                   indicate it's not.
%  guidance       - Source image for the affinity matrix. Same dimensions
%                   as the input image IN. Default: log(IN)
%  lambda         - Balances between the data term and the smoothness
%                   term. Increasing lambda will produce smoother images.
%                   Default value is 0.05 
%
% This function is based on the implementation of the WLS Filer by Farbman,
% Fattal, Lischinski and Szeliski, "Edge-Preserving Decompositions for 
% Multi-Scale Tone and Detail Manipulation", ACM Transactions on Graphics, 2008
% The original function can be downloaded from: 
% http://www.cs.huji.ac.il/~danix/epd/wlsFilter.m


small_num = 0.00001;

if ~exist('lambda','var') || isempty(lambda), lambda = 0.05; end

[h,w,~] = size(guidance);
k = h*w;
guidance = rgb2gray(guidance);

% Compute affinities between adjacent pixels based on gradients of guidance
dy = diff(guidance, 1, 1);
dy = -lambda./(sum(abs(dy).^2,3) + small_num);
dy = padarray(dy, [1 0], 'post');
dy = dy(:);

dx = diff(guidance, 1, 2); 
dx = -lambda./(sum(abs(dx).^2,3) + small_num);
dx = padarray(dx, [0 1], 'post');
dx = dx(:);


% Construct a five-point spatially inhomogeneous Laplacian matrix
B = [dx, dy];
d = [-h,-1];
tmp = spdiags(B,d,k,k);

ea = dx;
we = padarray(dx, h, 'pre'); we = we(1:end-h);
so = dy;
no = padarray(dy, 1, 'pre'); no = no(1:end-1);

D = -(ea+we+so+no);
Asmoothness = tmp + tmp' + spdiags(D, 0, k, k);

% Normalize data weight
data_weight = data_weight - min(data_weight(:)) ;
data_weight = 1.*data_weight./(max(data_weight(:))+small_num);

% Make sure we have a boundary condition for the top line:
% It will be the minimum of the transmission in each column
% With reliability 0.8
reliability_mask = data_weight(1,:) < 0.6; % find missing boundary condition
in_row1 = min( in,[], 1);
data_weight(1,reliability_mask) = 0.8;
in(1,reliability_mask) = in_row1(reliability_mask);

Adata = spdiags(data_weight(:), 0, k, k);

A = Adata + Asmoothness;
b = Adata*in(:);

% Solve
% out = lsqnonneg(A,b);
out = A\b;
out = reshape(out, h, w);
