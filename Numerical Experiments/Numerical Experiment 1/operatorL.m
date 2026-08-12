function At=operatorL(A,opt)

    if isstring(opt)
        opt = char(opt);
    end
  
    if strcmpi(opt,'t')
        At=fft(A,[],3);
    elseif strcmpi(opt,'c')
        At=dct(A,[],3);
    elseif strcmpi(opt,'f')
        At=A;
    elseif isnumeric(opt) && ismatrix(opt) && all(size(opt) >= [2 2])
        M=opt;
        if size(A,3)~=size(M,2)
            error('The dimension of the L-operator matrix does not match the third dimension of tensors A and B.');
        end   
        At=mode3_product(A,M);
    else
        error("The third input must be 't' (t-product), 'c' (reduced c-product), 'f' (facewise product), or a matrix M.")    
    end

end