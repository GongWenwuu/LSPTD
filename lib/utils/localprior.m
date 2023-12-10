function [L, T, gamma, flag] = localprior(X, Usize, Para, prior)
% Compute constant matrices which serve as the role of characterizing the feature of the X
% Inputs:
% X - input data
% Para - boolean or array indicats the spatiotemporal form
% prior - type of graph Laplacian to use ('stdc' or 'lrstd' or 'identity') for spatio property
%
% Outputs:
% gamma - regularization parameters
% L - cell array of Laplacian matrices
% T - cell array of transform matrices

N = ndims(X);
gamma = zeros(1, N);
L = cell(1, N);
T = cell(1, N);

if isempty(Para)
    flag = 2.*ones(1, N);
else
    flag = Para;
    for n = 1:N
        Xmat = reshape(permute(X, [n 1:n-1 n+1:N]), size(X, n), []);
        if flag(n) == 1
            if strcmp(prior, 'stdc')
                L = constructL_stdc(size(X), {1, 2, 3}, 2, L);
            elseif strcmp(prior, 'lrstd')
                L = constructL_lrstd(X, 5, 0);
            elseif strcmp(prior, 'identity')
                L{n} = eyes(size(X, n));
                gamma(n) = 1/N;
            end
            gamma(n) = norm(Xmat, 2)/(2*norm(L{n}, 2));
        elseif flag(n) == 0
            if strcmp(prior, 'identity')
                T{n} = eye(Usize(n));
                gamma(n) = 1/N;
            else
                T{n} = constructT(Usize(n));
                gamma(n) = norm(Xmat, 2)/(2*norm(T{n}*T{n}', 2));
            end
        else
            error('Para and flag are wrong!');
        end
    end
end

end
