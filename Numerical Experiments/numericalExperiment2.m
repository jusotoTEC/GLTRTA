%% Generalized Low-Tubal-Rank Tensor Approximation under the M-Product
% Numerical Experiment 2: Complex-Valued Tensors, associated with the paper:
% "A Note on a Closed-Form Solution to the Generalized Low-Tubal-Rank Tensor Approximation Problem"
%
% Created by: Pablo Soto-Quiros

clc; clearvars; close all; rng(1);

%% Problem dimensions and parameters

p = 100; q = 200; m = 150; n = 250; s = 50;


% t-product: L(D) = fft(D,[],3) for every tensor D.
productType = 't';

% The identity transform is scaled unitary, with alpha^2 = s.
alpha = sqrt(s);

%% Input tensors

A = randn(p, q, s)+1i*randn(p, q, s);
B = randn(p, m, s)+1i*randn(p, m, s);
C = randn(n, q, s)+1i*randn(n, q, s);

%% Transform-domain representation

At = operatorL(A, productType);
Bt = operatorL(B, productType);
Ct = operatorL(C, productType);

%% Eigenvalues appearing in the minimum-error formula

maximumRank = min(m, n);
eigenvaluesT = zeros(maximumRank, s);

for k = 1:s

    Ak = At(:, :, k);
    Bk = Bt(:, :, k);
    Ck = Ct(:, :, k);

    % Matrix T_k from Theorem 3.1:

    Tk = pinv(Bk) * Ak * pinv(Ck) * Ck * Ak' * Bk;

    lambda = sort(real(eig(Tk)), 'descend');

    % Remove small negative values caused by numerical roundoff.
    eigenvaluesT(:, k) = max(lambda, 0);

end

%% Minimum approximation error for each admissible tubal rank

normA2 = normFrob3d(A)^2;
tubalRanks = (1:maximumRank).';

cumulativeEigenvalues = cumsum(eigenvaluesT, 1);

minimumError = normA2 - alpha^(-2) * sum(cumulativeEigenvalues, 2);

% Remove small negative errors caused by floating-point roundoff.
minimumError = max(minimumError, 0);

%% Plot of the minimum approximation error

figure('Position', [100, 100, 1000, 250]);

plot(tubalRanks, minimumError, '-', 'LineWidth', 2, 'MarkerSize', 8);

xlabel('Tubal rank (r)',  'FontSize', 12);

ylabel('Minimum error',  'FontSize', 12);

xlim([1, maximumRank]);

grid on;
box on;

