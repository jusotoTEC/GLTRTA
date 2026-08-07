%% Generalized Low-Tubal-Rank Tensor Approximation under the M-Product
% Numerical Experiment 1: Real-Valued Tensors, associated with the paper:
% "A Note on a Closed-Form Solution to the Generalized Low-Tubal-Rank Tensor Approximation Problem"
%
% Created by: Pablo Soto-Quiros

clc; clearvars; close all;

%% Problem dimensions and parameters

p = 5; q = 5; m = 4; n = 4; s = 2;

selectedRank = 2;

% Facewise product: L(D) = D for every tensor D.
productType = 'f';

% Since the identity transform is unitary, alpha = 1.
alpha = 1;

%% Input tensors

A = zeros(p, q, s);
B = zeros(p, m, s);
C = zeros(n, q, s);

A(:, :, 1) = [...
    -1,  1,  7,  3,  6;
    -4, -2,  6,  2,  6;
     3,  3,  1,  1,  0;
    -2, -4, -8, -4, -6;
    -1,  1,  7,  3,  6];

A(:, :, 2) = [...
     6,  6,  3, -2,  6;
     8,  8,  4, -8, -8;
     6,  6,  3, -4,  0;
     0,  0,  0,  0,  0;
    -8, -8, -4,  5, -1];

B(:, :, 1) = [...
     1, -2, -1,  1;
     1, -1, -2,  1;
     0,  0, -1, -1;
    -1,  2, -2, -2;
     1, -2,  1,  2];

B(:, :, 2) = [...
     1, -1,  0, -2;
     0,  0, -2,  0;
     1,  0,  0,  1;
     0, -1, -1, -1;
    -1, -2, -2, -2];

C(:, :, 1) = [...
    -2, -1, -1,  1, -1;
    -1, -2, -2, -2, -1;
    -2, -1,  1,  1,  1;
     1, -1,  0,  2, -2];

C(:, :, 2) = [...
    -1,  0,  0, -1,  1;
    -2, -2, -1,  2,  2;
     0, -1,  1, -1, -1;
     2, -2,  2,  1, -1];

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
    %
    % T_k = B_k^\dagger A_k C_k^\dagger C_k
    %       A_k^H B_k.

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

plot(tubalRanks, minimumError, '-o', 'LineWidth', 2, 'MarkerSize', 8);

xlabel('Tubal rank (r)',  'FontSize', 12);

ylabel('Minimum error',  'FontSize', 12);

xlim([1, maximumRank]);
xticks(tubalRanks);

grid on;
box on;

%% Closed-form optimal factors for the selected tubal rank

Xt = zeros(m, selectedRank, s);
Yt = zeros(selectedRank, n, s);

for k = 1:s

    Ak = At(:, :, k);
    Bk = Bt(:, :, k);
    Ck = Ct(:, :, k);

    % Projected matrix P_k.
    Pk = Bk * pinv(Bk) * Ak * pinv(Ck) * Ck;

    [Uk, Sk, Vk] = svd(Pk, 'econ');

    Ur = Uk(:, 1:selectedRank);
    Sr = Sk(1:selectedRank, 1:selectedRank);
    Vr = Vk(:, 1:selectedRank);

    % Optimal factors in the transform domain.
    Xt(:, :, k) = pinv(Bk) * Ur;
    Yt(:, :, k) = Sr * Vr' * pinv(Ck);

end

%% Optimal tensors in the original domain

X = operatorLinv(Xt, productType);
Y = operatorLinv(Yt, productType);

Z = mprod(X, Y, productType);

%% Verification of the approximation error

approximation = mprod(mprod(B, Z, productType), C, productType);

computedError = normFrob3d(A - approximation)^2;
predictedError = minimumError(selectedRank);

fprintf('Selected tubal rank: r = %d\n', selectedRank);
fprintf('Predicted minimum error: %.10e\n', predictedError);
fprintf('Computed approximation error: %.10e\n', computedError);
fprintf('Absolute difference: %.10e\n', abs(predictedError - computedError));

%% Display the optimal factors

disp('Optimal tensor X:');
disp(X);

disp('Optimal tensor Y:');
disp(Y);