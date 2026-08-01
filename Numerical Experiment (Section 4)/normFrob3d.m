function frobNorm = normFrob3d(A)
%normFrob3d Compute the Frobenius norm of a third-order tensor.
%
%   frobNorm = normFrob3d(A) returns the Frobenius norm of the
%   third-order tensor A.
%
%   Input
%   -----
%   A : Third-order tensor.
%
%   Output
%   ------
%   frobNorm : Frobenius norm of A.
%
%   Created by: Pablo Soto-Quiros

    frobNorm = sqrt(sum(A.^2, 'all'));

end