function T = constructT(dim)

    c = zeros(1,dim);c(1) = 1;
    r = zeros(1,dim-1);r(1) = 1;r(2) = -1;
    T = toeplitz(c,r);

end