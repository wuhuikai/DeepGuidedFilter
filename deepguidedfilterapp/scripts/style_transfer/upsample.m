% Upsampling procedure.
%
% Argments:
%   'I': image
%   'filter': 2D separable upsampling filter
%   parent subwindow indices 'subwindow', given as [r1 r2 c1 c2]
%
% tom.mertens@gmail.com, August 2007
% sam.hasinoff@gmail.com, March 2011  [handle subwindows, reweighted boundaries]
%

function R = upsample(I, filter, subwindow)

% increase size to match dimensions of the parent subwindow, 
% about 2x in each dimension
r = subwindow(2) - subwindow(1) + 1;
c = subwindow(4) - subwindow(3) + 1;
k = size(I,3);
reven = mod(subwindow(1),2)==0;
ceven = mod(subwindow(3),2)==0;

border_mode = 'reweighted';
%border_mode = 'symmetric';

switch border_mode
    case 'reweighted'        
        % interpolate, convolve with 2D separable filter
        R = zeros(r,c,k);
        R(1+reven:2:r, 1+ceven:2:c, :) = I;
        R = imfilter(R,filter);
        
        % reweight, brute force weights from 1's in valid image positions
        Z = zeros(r,c,k);
        Z(1+reven:2:r, 1+ceven:2:c, :) = 1;
        Z = imfilter(Z,filter);
        R = R./Z;
        
    otherwise
        % increase resolution
        I = padarray(I,[1 1 0],'replicate'); % pad the image with a 1-pixel border
        R = zeros(r+4,c+4,k);
        R(1+reven:2:end, 1+ceven:2:end, :) = 4*I;
        
        % interpolate, convolve with 2D separable filter
        R = imfilter(R,filter,border_mode);
        
        % remove the border
        R = R(3:end-2, 3:end-2, :);     
end
end