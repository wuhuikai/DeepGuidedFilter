function [O]=llf_discrete(I,f,N)
% apply the llf using a discrete transfer function
% input:
% -I the image in [0 1]
% -f the transfer function given as a vector of size 256 corresponding to the mapping of [0 1]
% -N levels of doscretization for the llf. 

[height width]=size(I);
n_levels=ceil(log(min(height,width))-log(2))+2;
discretisation=linspace(0,1,N);
discretisation_step=discretisation(2);

input_gaussian_pyr=gaussian_pyramid(I,n_levels);
output_laplace_pyr=laplacian_pyramid(I,n_levels);
output_laplace_pyr{n_levels}=input_gaussian_pyr{n_levels};
f(257)=1;
for ref=discretisation
    index=floor(abs(255.*(I-ref)));
    x=min((abs(255.*(I-ref))-index)./255,255);
    I_remap=sign(I-ref).*((1-x).*f(index+1)+(x).*f(index+2));%fact*(I-ref).*exp(-(I-ref).*(I-ref)./(2*sigma*sigma));
    temp_laplace_pyr=laplacian_pyramid(I_remap,n_levels);
    for level=1:n_levels-1
        output_laplace_pyr{level}=output_laplace_pyr{level}+...
            (abs(input_gaussian_pyr{level}-ref)<discretisation_step).*...
            temp_laplace_pyr{level}.*...
            (1-abs(input_gaussian_pyr{level}-ref)/discretisation_step);
    end
end

O=reconstruct_laplacian_pyramid(output_laplace_pyr);
end
