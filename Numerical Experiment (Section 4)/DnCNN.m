%% DnCNN Image Denoising
% Numerical experiments associated with the paper:
% "Generalized Low-Tubal-Rank Tensor Approximation under the M-Product"
%
% This script:
%   1. Loads the pre-trained DnCNN denoising network.
%   2. Removes noise from an RGB image.
%   3. Computes the average SSIM with respect to the source image.
%
% Created by: Pablo Soto-Quiros

clc; clear; close all;


%% Load the pre-trained DnCNN network

net = denoisingNetwork("dncnn");

%% Load the source and noisy images

sourceImage = im2double(imread(fullfile('example','source.jpg')));
noisyImage  = im2double(imread(fullfile('example','noisy.jpg')));

%% Split the noisy image into RGB channels

[redChannel, greenChannel, blueChannel] = imsplit(noisyImage);

%% Apply the DnCNN denoising network to each channel

denoisedRed   = denoiseImage(redChannel, net);
denoisedGreen = denoiseImage(greenChannel, net);
denoisedBlue  = denoiseImage(blueChannel, net);

%% Reconstruct the denoised RGB image

cleanedImage = cat(3, denoisedRed, denoisedGreen, denoisedBlue);

%% Display the reconstructed image

figure;
imshow(cleanedImage);

%% Compute the average SSIM

ssimChannels = zeros(3,1);

for k = 1:3
    ssimChannels(k) = ssim(sourceImage(:,:,k), cleanedImage(:,:,k));
end

ssimValue = mean(ssimChannels);

%% Display the reconstruction results

fprintf('\n===============================================\n');
fprintf('DnCNN Image Denoising Results\n');
fprintf('Average SSIM : %.4f\n', ssimValue);
fprintf('===============================================\n');