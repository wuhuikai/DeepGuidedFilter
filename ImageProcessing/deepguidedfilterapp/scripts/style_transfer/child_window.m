function child = child_window(parent,N)
% for a parent subwindow [r1 r2 c1 c2], find the corresponding
% child subwindow at the coarser pyramid level N levels up

if ~exist('N','var')
    N = 1;
end

child = parent;
for K = 1:N
    child = (child+1)/2;
    child([1 3]) = ceil(child([1 3]));
    child([2 4]) = floor(child([2 4]));
end

end