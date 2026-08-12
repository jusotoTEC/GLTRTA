function x=normFrob3d(A)
    x = sqrt(sum(abs(A).^2,'all'));
end