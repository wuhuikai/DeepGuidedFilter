function O=apply_tranfer_function(I,f)
% apply a transfer function 
% intput:
% -I is the image in [0 1] to which apply the transfer function
% -f is the transfer function given as a vector of size 256
index=max(min(floor(255.*I),255),0);
x=min((255.*I-index)./255,255);
f(257)=1;
O=((1-x).*f(index+1)+(x).*f(index+2));
end
