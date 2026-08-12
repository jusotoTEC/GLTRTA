function C=mprod(A,B,opt)

%   mprod: Tensor-tensor product along the third mode with multiple options.
%
%   C = mprod(A,B) multiplies tensors A (m1×n1×p) and B (m2×n2×p)
%   using the t-product (FFT along the 3rd dimension). The result C has
%   size (m1×n2×p). This is the default option if OPT is not provided.
%
%   C = mprod(A,B,opt) specifies the type of product:
%       opt = 't'  — t-product based on FFT along the 3rd dimension
%                    (default if OPT is omitted).
%       opt = 'c'  — reduced c-product based on DCT along the 3rd dimension.
%       opt = 'f'  — facewise product: multiplies each frontal slice.
%        opt = M   — an invertible matrix (p×p) used as a linear transform for a
%                    generalized L-product.
%
%   INPUTS
%       A   — tensor of size m1×q×p (numeric).
%       B   — tensor of size q×n2×p (numeric).
%       opt — (optional) product selector:
%                * character/string: 't', 'c', or 'f'
%                * numeric matrix M of size p×p
%
%   OUTPUT
%       C   — resulting tensor:
%                * If opt is 't', 'c', or 'f':  size(C) = [m1, n2, p].
%                * If opt is a matrix M (p×p): size(C) = [m1, n2, p].
%
%   DIMENSION REQUIREMENTS
%       - size(A,2) must equal size(B,1).
%       - size(A,3) must equal size(B,3).
%       - If opt is a matrix M (q×p), then size(M,2) = size(A,3).
%
%   EXAMPLES
%       % Default t-product
%       A = rand(4,3,10); B = rand(3,5,10);
%       C = mprod(A,B);                      % C: 4×5×10
%
%       % c-product
%       Cc = mprod(A,B,'c');                 % Cc: 4×5×10
%
%       % facewise product
%       Cf = mprod(A,B,'f');                 % Cf: 4×5×10
%
%       % L-product with a transform matrix M (q×p)
%       p = 10; q = 6; M = randn(p,p);
%       CL = mprod(A,B,M);                   % CL: 4×5×6
%
%   NOTES
%       - Requires MATLAB R2020b or later for PAGEMTIMES.
%
%   See also fft, ifft, dct, idct, pagemtimes.


    [m1,n1,p1]=size(A);
    [m2,n2,p2]=size(B);
    
    if or(n1~=m2,p1~=p2)
        error('Inner tensor dimensions must agree.')
    end
    
    Ct=zeros(m1,n2,p1);
    
    if nargin==2
        opt='t';
    elseif nargin<=1
        error('This command requires at least two input arguments: tensors A and B.');
    end


    if isstring(opt)
        opt = char(opt);
    end


    if strcmpi(opt,'t')
        At=fft(A,[],3);
        Bt=fft(B,[],3);
        halfp1=ceil((p1+1)/2);
        Ct(:,:,1:halfp1)=pagemtimes(At(:,:,1:halfp1),Bt(:,:,1:halfp1));
        idx = (halfp1+1):p1;
        mirror = p1 + 2 - idx;
        Ct(:,:,idx) = conj(Ct(:,:,mirror));
        C=real(ifft(Ct,[],3));
    elseif strcmpi(opt,'c')
        At=dct(A,[],3);
        Bt=dct(B,[],3);
        Ct=pagemtimes(At,Bt);
        C=idct(Ct,[],3);
    elseif strcmpi(opt,'f')
        C=pagemtimes(A,B);
    elseif isnumeric(opt) && ismatrix(opt) && all(size(opt) >= [2 2])
        M=opt;
        n3=size(M,2);
        if n3~=p1
            error('The dimension of the L-operator matrix does not match the third dimension of tensors A and B.');
        end   
        At=mode3_product(A,M);
        Bt=mode3_product(B,M);
        Ct=pagemtimes(At,Bt);
        C=mode3_product(Ct,M^-1);
    else
        error("The third input must be 't' (t-product), 'c' (reduced c-product), 'f' (facewise product), or a matrix M.")    
    end
end