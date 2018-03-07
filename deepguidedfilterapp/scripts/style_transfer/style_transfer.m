function [Oi Og]=style_transfer(I,M,N,n_iterations)
% transfer style from black and white images
% input:
% -I is the input image, M the model
% -N is the number reference intensities used for the LLF
% -n_iterations is the number of iterations of the transfer
% output:
% -Og is the output finihing with a gradient transfer
% -Oi is the output finishing with an intensity transfer
    if nargin<4
       n_iterations=5;
    end
    if nargin<3
       N=20;
    end
    GM=sqrt((M(1:end-1,1:end-1)-M(2:end,1:end-1)).^2+(M(1:end-1,1:end-1)-M(1:end-1,2:end)).^2);
    f=get_transfer_function(I,M);
    Oi=apply_tranfer_function(I,f);
    for t=1:n_iterations
        fprintf('iteration %i ...\n',t);
        GI=sqrt((Oi(1:end-1,1:end-1)-Oi(2:end,1:end-1)).^2+(Oi(1:end-1,1:end-1)-Oi(1:end-1,2:end)).^2);
        f=get_transfer_function(GI,GM);
        Og=llf_discrete(Oi,f,N);
        Oi=(Og-min(Og(:)))./(max(Og(:))-min(Og(:)));
        f=get_transfer_function(Oi,M);
        Oi=apply_tranfer_function(Oi,f);
    end
    O=I;
end
