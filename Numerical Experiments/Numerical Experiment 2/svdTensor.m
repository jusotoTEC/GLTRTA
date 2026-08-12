function [U,S,V]=svdTensor(A,opt)

    if isstring(opt)
        opt = char(opt);
    end

    [m,n,p]=size(A);
  
    if strcmpi(opt,'t')
        At=fft(A,[],3);
    elseif strcmpi(opt,'c')
        At=dct(A,[],3);
    elseif strcmpi(opt,'f')
        At=A;
    elseif isnumeric(opt) && ismatrix(opt) && all(size(opt) >= [2 2])
        M=opt;
        p1=size(M,2);
        if p~=p1
            error('The dimension of the L-operator matrix does not match the third dimension of tensors A and B.');
        end   
        At=mode3_product(A,M);
    else
        error("The third input must be 't' (t-product), 'c' (reduced c-product), 'f' (facewise product), or a matrix M.")    
    end

    Ut=zeros(m,m,p); St=zeros(m,n,p); Vt=zeros(m,n,p);

    for k=1:p
        [Ut(:,:,k),St(:,:,k),Vt(:,:,k),]=svd(At(:,:,k));
    end

    if strcmpi(opt,'t')
        U=ifft(Ut,[],3); S=ifft(St,[],3); V=ifft(Vt,[],3);
    elseif strcmpi(opt,'c')
        U=idct(Ut,[],3); S=idct(St,[],3); V=idct(Vt,[],3);
    elseif strcmpi(opt,'f')
        U=Ut; S=St; V=Vt; 
    elseif isnumeric(opt) && ismatrix(opt) && all(size(opt) >= [2 2])
        U=mode3_product(Ut,M^-1);
        S=mode3_product(St,M^-1);
        V=mode3_product(Vt,M^-1);
    end
end