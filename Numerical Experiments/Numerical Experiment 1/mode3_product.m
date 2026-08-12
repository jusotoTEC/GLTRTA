function D=mode3_product(C,M)
    [m,n,p]=size(C);
    C2 = reshape(C, [m*n, p])';       
    D2 = M * C2;     
    D = reshape(D2', [m, n, size(M,1)]);
end 