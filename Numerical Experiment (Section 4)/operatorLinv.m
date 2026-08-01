function At = operatorLinv(A, opt)
%operatorLinv Apply the inverse L-operator to a third-order tensor.
%
%   At = operatorLinv(A, opt) applies the inverse transform associated
%   with the specified tensor-tensor product.
%
%   Input
%   -----
%   A   : Third-order tensor.
%   opt : Transform type:
%         't' - t-product (inverse FFT)
%         'c' - Reduced c-product (inverse DCT-II)
%         'f' - Facewise product (identity transform)
%          M  - Invertible transformation matrix.
%
%   Output
%   ------
%   At : Tensor after applying the inverse L-operator.
%
%   Created by: Pablo Soto-Quiros

    if isstring(opt)
        opt = char(opt);
    end

    if strcmpi(opt,'t')

        At = ifft(A,[],3);

    elseif strcmpi(opt,'c')

        At = idct(A,[],3);

    elseif strcmpi(opt,'f')

        At = A;

    elseif isnumeric(opt) && ismatrix(opt) && all(size(opt) >= [2 2])

        M = opt;

        if size(A,3) ~= size(M,2)
            error(['The dimension of the transformation matrix M ' ...
                   'must match the third dimension of tensor A.']);
        end

        At = mode3_product(A, inv(M));

    else

        error(['The second input must be ''t'' (t-product), ' ...
               '''c'' (reduced c-product), ''f'' (facewise product), ' ...
               'or an invertible transformation matrix M.']);

    end

end