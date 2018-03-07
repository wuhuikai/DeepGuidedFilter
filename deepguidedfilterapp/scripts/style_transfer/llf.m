% Perform the local laplacian filter using the function 
% f(x,ref)=x+fact*(I-ref)*exp(-(I-ref)²/(2*sigma²))

% Perform the local laplacian filter using any function

% This script implements edge-aware detail and tone manipulation as 
% described in : 
% Fast and Robust Pyramid-based Image Processing. 
% Mathieu Aubry, Sylvain Paris, Samuel W. Hasinoff, Jan Kautz, and Frédo Durand. 
% MIT technical report, November 2011

% INPUT
% I : input greyscale image
% r : a function handle to the remaping function
% N : number discretisation values of the intensity
%

% OUTPUT
% F : filtered image

% aubry.mathieu@gmail.com Sept 2012


function [F]=llf(I,sigma,fact,N)

    [height width]=size(I);
    n_levels=ceil(log(min(height,width))-log(2))+2;
    discretisation=linspace(0,1,N);
    discretisation_step=discretisation(2);
    
    input_gaussian_pyr=gaussian_pyramid(I,n_levels);
    output_laplace_pyr=laplacian_pyramid(I,n_levels);
    output_laplace_pyr{n_levels}=input_gaussian_pyr{n_levels};
    
    for ref=discretisation
        I_remap=fact*(I-ref).*exp(-(I-ref).*(I-ref)./(2*sigma*sigma));
        temp_laplace_pyr=laplacian_pyramid(I_remap,n_levels);
        for level=1:n_levels-1
            output_laplace_pyr{level}=output_laplace_pyr{level}+...
                (abs(input_gaussian_pyr{level}-ref)<discretisation_step).*...
                temp_laplace_pyr{level}.*...
                (1-abs(input_gaussian_pyr{level}-ref)/discretisation_step);            
        end
    end
    
    F=reconstruct_laplacian_pyramid(output_laplace_pyr);
    
    
    
    
