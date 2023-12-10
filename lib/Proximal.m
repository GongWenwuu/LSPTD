function X = Proximal(Xk,grad,Lip,alpha,Opts)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = arg min { alpha*|| X ||_Opts + <grad(Xk),X-Xk> + (Lip/2)*|| X-Xk ||_F }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if strcmp(Opts,'tr')
    [U,S,V] = svd(Xk-grad/(Lip),'econ');
    S = diag(S);
    rank = sum(S > alpha/Lip);
    X = U(:,1:rank)*diag(S(1:rank)-alpha/Lip)*V(:,1:rank)';
elseif strcmp(Opts,'l1')
    X = thresholding(Xk-grad/(Lip), alpha/Lip);
else
    error('Parameter is wrong');
end

end

function X = thresholding(Z, tau)

    A_sign = sign(Z);
    X = abs(Z) - tau;
    X(X < 0) = 0;
    X = X .* A_sign;
    
end