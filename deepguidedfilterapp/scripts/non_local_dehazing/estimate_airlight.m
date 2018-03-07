function [ Aout ] = estimate_airlight( img, Amin, Amax, N, spacing, K, thres )
%Estimate airlight of an image, using a 3*2D Hough transform, where each
%point votes for a given location using a fixed set of angles.
%
%   This is an implementation of our paper:
%   Dana Berman, Tali Treibitz, Shai Avidan,
%   "Air-light Estimation using Haze-Lines", ICCP 2017
%   
%   Input arguments:
%   img - input image (mandatory)
%   Amin (Amax) - minimal (maximal) value if the air-light. Optional
%               Can be used to reduce the search space and save time
%               Either a scalar (identical for all color channels) or a 3
%               value vector (different range for each color channel)
%   N - number of colors clusters to use (the image is converted to an
%       indexed image of at most N different cluster). Optional
%   spacing - air-light candidates' resolution, 1/M in the paper. Optional
%   K - angular resolution. Optional
%   thres - cone resolution, optional, default recommended


%% Verify input params, set defaults when necessary (same as published results)
if ~exist('thres','var') || isempty(thres), thres = 0.01 ; end;
if ~exist('spacing','var') || isempty(spacing), spacing = 0.02 ; end; %1/M in the paper
if ~exist('n_colors','var') || isempty(N), N = 1000 ; end; %number of colors clusters
if ~exist('K','var') || isempty(K), K = 40 ; end; %number of angles

% Define search range for the air-light. The search range is different for each 
% color channel. These values were used in all of our experiments.
if ~exist('Amin','var') || isempty(Amin), Amin = [0,0.05,0.1]; end;
if ~exist('Amax','var') || isempty(Amax), Amax = 1; end;

% Air-light search range, accept a scalar if identical for all color channels
if isscalar(Amin), Amin = repmat(Amin,1,3); end 
if isscalar(Amax), Amax = repmat(Amax,1,3); end

%% Convert input image to an indexed image
[img_ind, points] = rgb2ind(img, N);
[h,w,~] = size(img);
% Remove empty clusters
idx_in_use = unique(img_ind(:));
idx_to_remove = setdiff(0:(size(points,1)-1),idx_in_use);
points(idx_to_remove+1,:) = [];
img_ind_sequential = zeros(h,w);
for kk = 1:length(idx_in_use)
    img_ind_sequential(img_ind==idx_in_use(kk)) = kk;
end
% Now the min value of img_ind_sequential is 1 rather then 0, and the indices
% correspond to points

% Count the occurences if each index - this is the clusters' weight
[points_weight,~] = histcounts(img_ind_sequential(:),size(points,1));
points_weight = points_weight./(h*w);
if ~ismatrix(points), points = reshape(points,[],3); end % verify dim

%% Define arrays of candidate air-light values and angles
angle_list = reshape(linspace(0, pi, K),[],1);
% Use angle_list(1:end-1) since angle_list(end)==pi, which is the same line
% in 2D as since angle_list(1)==0
directions_all = [sin(angle_list(1:end-1)) , cos(angle_list(1:end-1)) ];

% Air-light candidates in each color channel
ArangeR = Amin(1):spacing:Amax(1);
ArangeG = Amin(2):spacing:Amax(2);
ArangeB = Amin(3):spacing:Amax(3);

%% Estimate air-light in each pair of color channels
% Estimate RG
Aall = generate_Avals(ArangeR, ArangeG);
[~, AvoteRG] = vote_2D(points(:,1:2), points_weight, directions_all, Aall, thres );
% Estimate GB
Aall = generate_Avals(ArangeG, ArangeB);
[~, AvoteGB] = vote_2D(points(:,2:3), points_weight, directions_all, Aall, thres );
% Estimate RB
Aall = generate_Avals(ArangeR, ArangeB);
[~, AvoteRB] = vote_2D(points(:,[1,3]), points_weight, directions_all, Aall, thres);

%% Find most probable airlight from marginal probabilities (2D arrays)
% Normalize (otherwise the numbers are quite large)
max_val = max( [max(AvoteRB(:)) , max(AvoteRG(:)) , max(AvoteGB(:)) ]);
AvoteRG2 = AvoteRG./max_val;
AvoteGB2 = AvoteGB./max_val;
AvoteRB2 = AvoteRB./max_val;
% Generate 3D volumes from 3 different 2D arrays
A11 = repmat( reshape(AvoteRG2, length(ArangeG),length(ArangeR))', 1,1,length(ArangeB));
tmp = reshape(AvoteRB2, length(ArangeB),length(ArangeR))';
A22 = repmat(reshape(tmp, length(ArangeR),1,length(ArangeB)) , 1,length(ArangeG),1);
tmp2 = reshape(AvoteGB2, length(ArangeB),length(ArangeG))';
A33 = repmat(reshape(tmp2, 1, length(ArangeG),length(ArangeB)) , length(ArangeR),1,1);
AvoteAll = A11.*A22.*A33;
[~, idx] = max(AvoteAll(:));
[idx_r,idx_g,idx_b] = ind2sub([length(ArangeR),length(ArangeG),length(ArangeB)],idx);
Aout = [ArangeR(idx_r), ArangeG(idx_g), ArangeB(idx_b)];


end % function estimate_airlight_2D

%% Sub functions

function Aall = generate_Avals(Avals1, Avals2)
%Generate a list of air-light candidates of 2-channels, using two lists of
%values in a single channel each
%Aall's length is length(Avals1)*length(Avals2)
Avals1 = reshape(Avals1,[],1);
Avals2 = reshape(Avals2,[],1);
A1 = kron(Avals1, ones(length(Avals2),1));
A2 = kron(ones(length(Avals1),1), Avals2);
Aall = [A1, A2];
end % function generate_Avals

function [Aout, Avote2] = vote_2D(points, points_weight, directions_all, Aall, thres)
n_directions = size(directions_all,1);
accumulator_votes_idx = false(size(Aall,1), size(points,1), n_directions);
for i_point = 1:size(points,1)
    for i_direction = 1:n_directions
		 % save time and ignore irelevant points from the get-go
        idx_to_use = find( (Aall(:, 1) > points(i_point, 1)) & (Aall(:, 2) > points(i_point, 2)));
        if isempty(idx_to_use), continue; end
		
        % calculate distance between all A options and the line defined by
        % i_point and i_direction. If the distance is smaller than a thres,
        % increase the cell in accumulator
        dist1 = sqrt(sum([Aall(idx_to_use, 1)-points(i_point, 1), Aall(idx_to_use, 2)-points(i_point, 2)].^2,2));
        %dist1 = dist1 - min(dist1);
        dist1 = dist1./sqrt(2) + 1;
        
        dist =  -points(i_point, 1)*directions_all(i_direction,2) + ...
            points(i_point, 2)*directions_all(i_direction,1) + ...
            Aall(idx_to_use, 1)*directions_all(i_direction,2) - ...
            Aall(idx_to_use, 2)*directions_all(i_direction,1);
        idx = abs(dist)<2*thres.*dist1;
        if ~any(idx), continue; end

        idx_full = idx_to_use(idx);
        accumulator_votes_idx(idx_full, i_point,i_direction) = true;
    end
end
% use only haze-lined that are supported by 2 points or more
accumulator_votes_idx2 = (sum(uint8(accumulator_votes_idx),2))>=2; 
accumulator_votes_idx = bsxfun(@and, accumulator_votes_idx ,accumulator_votes_idx2);
accumulator_unique = zeros(size(Aall,1),1);
for iA = 1:size(Aall,1)
    idx_to_use = find(Aall(iA, 1) > points(:, 1) & (Aall(iA, 2) > points(:, 2)));
    points_dist = sqrt((Aall(iA,1) - points(idx_to_use,1)).^2+(Aall(iA,2) - points(idx_to_use,2)).^2);
    points_weight_dist = points_weight(idx_to_use).*(5.*exp(-reshape(points_dist,1,[]))+1); 
    accumulator_unique(iA) = sum(points_weight_dist(any(accumulator_votes_idx(iA,idx_to_use,:),3)));
end
[~, Aestimate_idx] = max(accumulator_unique);
Aout = Aall(Aestimate_idx,:);
Avote2 = accumulator_unique; 

end % function vote_2D
