function At = operatorL(A, opt)
%OPERATORL Apply the L-operator to a third-order tensor.
%
%   At = OPERATORL(A, opt) applies the transform associated with the
%   specified tensor-tensor product.
%
%   Input
%   -----
%   A   : Third-order tensor.
%   opt : Transform type:
%         't' - t-product (FFT)
%         'c' - Reduced c-product (DCT-II)
%         'f' - Facewise product (identity transform)
%         M   - Invertible transformation matrix.
%
%   Output
%   ------
%   At : Tensor after applying the L-operator.
%
%   Created by: Pablo Soto-Quiros

    if isstring(opt)
        opt = char(opt);
    end

    if strcmpi(opt,'t')

        At = fft(A, [], 3);

    elseif strcmpi(opt,'c')

        At = dct(A, [], 3);

    elseif strcmpi(opt,'f')

        At = A;

    elseif isnumeric(opt) && ismatrix(opt) && all(size(opt) >= [2 2])

        M = opt;

        if size(A,3) ~= size(M,2)
            error(['The dimension of the transformation matrix M ' ...
                   'must match the third dimension of tensor A.']);
        end

        At = mode3_product(A, M);

    else

        error(['The second input must be ''t'' (t-product), ' ...
               '''c'' (reduced c-product), ''f'' (facewise product), ' ...
               'or an invertible transformation matrix M.']);

    end

end