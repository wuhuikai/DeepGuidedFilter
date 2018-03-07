function f=get_transfer_function(I,M)
% input:
%   I : image in [0 1] with stat you want to be transfered
%   M : image in [0 1] with stat you want to get to
% output
%   f  : table with transfer values of size 256 and values in [0,1]

% get cdf of the input values
x=linspace(0,1,256);
h1=hist(I(:),x);
cdf1=cumsum(h1)./sum(h1);

% get cdf target values (more resolution is needed because we want to invert it)
x=linspace(0,1,256^2); 
h2=hist(M(:),x);
cdf2=cumsum(h2)./sum(h2);
f=linspace(0,1,256);

% get f=cdf2^{-1}(cdf1)
for i=1:256
    f(i)=find(cdf2>=cdf1(i),1)./(256*256);
end

end
