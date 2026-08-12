%% Generalized Low-Tubal-Rank Tensor Approximation under the M-Product
% Numerical experiments associated with the paper:
% "A Closed-Form Solution for Generalized Low-Tubal-Rank Tensor Approximationt"
%
% Created by: Pablo Soto-Quiros
%
% This script:
%   1. Constructs the training tensors from source and noisy images.
%   2. Computes the minimum approximation error for different tubal ranks.
%   3. Computes the optimal low-tubal-rank reconstruction operator.
%   4. Reconstructs a noisy image and evaluates its SSIM.

clc; clear; close all;

%% Parameters

imageSize = 128;
numImages = 1132;
numChannels = 3;
transformType = 'c';   % Type-II DCT (reduced c-product)
selectedRank = 800;    


%% Construct training tensors

A = zeros(imageSize^2, numImages, numChannels);
C = zeros(imageSize^2, numImages, numChannels);

for k = 1:numImages
    % Read and vectorize the source image
    sourcePath = fullfile('coast_128x128_source', ['c (', num2str(k), ').jpg']);
    sourceImage = im2double(imread(sourcePath));
    A(:, k, 1) = reshape(sourceImage(:, :, 1), [], 1);
    A(:, k, 2) = reshape(sourceImage(:, :, 2), [], 1);
    A(:, k, 3) = reshape(sourceImage(:, :, 3), [], 1);

    % Read and vectorize the noisy image
    noisyPath = fullfile('coast_128x128_noisy', ['c (', num2str(k), ').jpg']);
    noisyImage = im2double(imread(noisyPath));
    C(:, k, 1) = reshape(noisyImage(:, :, 1), [], 1);
    C(:, k, 2) = reshape(noisyImage(:, :, 2), [], 1);
    C(:, k, 3) = reshape(noisyImage(:, :, 3), [], 1);
end


%% Apply the tensor transform

At = operatorL(A, transformType);
Ct = operatorL(C, transformType);


%% Compute the eigenvalues associated with each frontal slice

eigTi = zeros(numImages, numChannels);

for k = 1:numChannels
    Atk = A(:, :, k);
    Ctk = C(:, :, k);
    Tk = Atk' * Atk * pinv(Ctk) * Ctk;
    eigTi(:, k) = sort(eig(Tk), 'descend');
end


%% Compute the minimum approximation error

normASquared = normFrob3d(A)^2;
tensorRankVector = 1:numImages;
errorVector = zeros(numImages, 1);
for r = tensorRankVector
    eigenvalueSum = 0;
    for k = 1:numChannels
        eigenvalueSum = eigenvalueSum + sum(eigTi(1:r, k));
    end
    errorVector(r) = normASquared - eigenvalueSum;
end


%% Plot the normalized minimum approximation error

figure('Position', [100, 100, 1000, 200]);
plot(tensorRankVector, errorVector / max(errorVector));
xlabel('Tensor rank (r)', 'FontSize', 12);
ylabel('Normalized error', 'FontSize', 12);
grid on;
xlim([1, numImages]);


%% Compute the optimal low-tubal-rank operator Z = X *_M Y

Xt = zeros(imageSize^2, selectedRank, numChannels);
Yt = zeros(selectedRank, imageSize^2, numChannels);

for k = 1:numChannels
    Pk = At(:, :, k) * pinv(Ct(:, :, k)) * Ct(:, :, k);
    [U, S, V] = svd(Pk);
    Xt(:, :, k) = U(:, 1:selectedRank);
    Yt(:, :, k) = ...
        S(1:selectedRank, 1:selectedRank) * ...
        V(:, 1:selectedRank)' * ...
        pinv(Ct(:, :, k));
end

X = operatorLinv(Xt, transformType);
Y = operatorLinv(Yt, transformType);


%% Reconstruct the noisy image

noisyImage = im2double(imread(fullfile('example', 'noisy.jpg')));
yNoisy = zeros(imageSize^2, 1, numChannels);

for k = 1:numChannels
    yNoisy(:, 1, k) = reshape(noisyImage(:, :, k), [], 1);
end

auxiliaryTensor = mprod(Y, yNoisy, transformType);
reconstructedTensor = mprod(X, auxiliaryTensor, transformType);
cleanedImage = reshape(reconstructedTensor, [imageSize, imageSize, numChannels]);


%% Display the reconstructed image

figure;
imshow(cleanedImage);


%% Compute the SSIM

sourceImage = im2double(imread(fullfile('example', 'source.jpg')));
ssimChannels = zeros(numChannels, 1);

for k = 1:numChannels
    ssimChannels(k) = ssim(sourceImage(:, :, k), cleanedImage(:, :, k));
end

ssimValue = mean(ssimChannels);

%% Results

fprintf('\n-----------------------------------------------\n');
fprintf('Image reconstruction completed successfully.\n');
fprintf('Selected tubal rank : %d\n', selectedRank);
fprintf('Average SSIM        : %.4f\n', ssimValue);
fprintf('-----------------------------------------------\n');
