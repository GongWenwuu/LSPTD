function w = myweight(X, sz, flag, alpha)

w = zeros(1,sz);
for n = 1:sz
    if flag == 0
        w(n) = factorweight(X, n, 0,[]);
    else
        w(n) = factorweight(X, n, alpha, flag);
    end
end
end

function w_n = factorweight(X, mode, alpha, Para)

if strcmp(Para,'lrstd')
    N = length(X);
    sValue = cell(1,N-1); Unorm = ones(1,N-1);
    indices = 1:N;
    indices(mode) = [];
    for n = indices
        sValue{n} = svd(X{n}, 'econ');
        Unorm(n)  = sum(sValue{n});
    end
    if alpha == 0
        w_n = prod(Unorm);
    else
        w_n = (1-alpha)*prod(Unorm)/alpha;
    end
elseif strcmp(Para,'prod')
    N = length(X);
    sValue = cell(1,N-1); Unormexcep = ones(1,N-1);
    indices = 1:N;
    indices(mode) = [];
    for n = indices
        sValue{n} = svd(X{n}, 'econ');
        Unormexcep(n)  = sum(sValue{n});
    end
    if alpha == 0
        w_n = prod(Unormexcep)/Usum(X);
    else
        w_n = (1-alpha)*prod(Unormexcep)/(alpha*Usum(X));
    end
elseif strcmp(Para,'sum')
    N = ndims(X);
    sigma = cell(1,N); Xnorm = 0;
    for n = 1:N
        Xn  = reshape(permute(X, [n 1:n-1 n+1:N]),size(X,n),[]);
        sigma{n} = svd(Xn, 'econ');
        Xnorm  = Xnorm + sum(sigma{n});
    end
    Xn  = reshape(permute(X, [mode 1:mode-1 mode+1:N]),size(X,mode),[]);
    if alpha == 0
        w_n = sum(svd(Xn, 'econ'))/Xnorm;
    else
        w_n = (1-alpha)*sum(svd(Xn, 'econ'))/(Xnorm*alpha);
    end
else
    if alpha == 0
        w_n = 1;
    else
        w_n = (1-alpha)/alpha;
    end
end

end

function Usum = Usum(U)
N = length(U); Unorm = ones(1,N);
for n = 1:N
    sigma = cell(1,N-1); temp = ones(1,N-1);
    indices = 1:N;
    indices(n) = [];
    for i = indices
        sigma{i} = svd(U{i}, 'econ');
        temp(i) = sum(sigma{i});
    end
    Unorm(n) = prod(temp);
end
Usum = sum(Unorm);
end