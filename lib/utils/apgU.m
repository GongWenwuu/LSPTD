function [Un, Lip] = apgU(U,G,X,Y,L,T,alpha,gamma,flag,mu,mode)
% \alpha\left\|\mathbf{U}_{n}\right\|_{*} + \frac{1}{2} \operatorname{tr}\left(\mathbf{U}_{n} \mu \mathbf{G}_{(n)} \mathbf{V}_{n}^{\mathrm{T}} \mathbf{V}_{n} \mathbf{G}_{(n)}^{\mathrm{T}} \mathbf{U}_{n}^{\mathrm{T}} \right) \\
% - \operatorname{tr}\left(\mathbf{U}_{n}^{\mathrm{T}} \left(\left(\mu\mathbf{X}_{(n)}+\mathbf{Y}_{(n)}\right) \mathbf{V}_{n} \mathbf{G}_{(n)}^{\mathrm{T}} \right)
%% Initialization
Usq = U; 
for n = 1:ndims(X)
    Usq{n} = U{n}'*U{n};
end
Uextra = U{mode};
%% Update by linearization
Bsq = mu*Matrixization(G,Usq,mode,'decompress')*Unfold(G,mode)';
XB = Matrixization(mu*X+Y,U,mode,'compress')*Unfold(G,mode)';
if flag == 1
    grad = Uextra*Bsq - XB + gamma*L*Uextra;
    Lip = norm(Bsq,2) + gamma*norm(L,2);
elseif flag == 0
    grad = Uextra*Bsq - XB + gamma*Uextra*(T*T');
    Lip = norm(Bsq,2) + gamma*norm((T*T'),2);
else
    grad = Uextra*Bsq - XB;
    Lip = norm(Bsq,2);
end

Un = Proximal(Uextra,grad,Lip,alpha,'tr');

end

function X_unf = Unfold(X,mode)
    N = ndims(X);
    X_unf = reshape(permute(X, [mode 1:mode-1 mode+1:N]),size(X,mode),[]);
end