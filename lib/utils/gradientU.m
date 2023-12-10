function grad = gradientU(Uextra, U, Usq, G, X, L, T, gamma, mode, flag)
% L_{\mathbf{U}_{n}}= \mathbf{U}_{n} \mathbf{G}_{(n)} \mathbf{V}_{n}^{\mathrm{T}}\mathbf{V}_{n} \mathbf{G}_{(n)}^{\mathrm{T}} 
% - \mathbf{X}_{(n)}\mathbf{V}_{n}\mathbf{G}_{(n)}^{\mathrm{T}} + \alpha \mathbf{L}_n\mathbf{U}_n
Bsq = Matrixization(G,Usq,mode,'decompress')*Unfold(G,mode)';
XB = Matrixization(X,U,mode,'compress')*Unfold(G,mode)';

if flag == 1
    grad = Uextra{mode}*Bsq - XB + gamma*L*Uextra{mode};
elseif flag == 0
    grad = Uextra{mode}*Bsq - XB + gamma*Uextra{mode}*(T*T');
else
    grad = Uextra{mode}*Bsq - XB;
end

end

function X_unf = Unfold(X,mode)
    N = ndims(X);
    X_unf = reshape(permute(X, [mode 1:mode-1 mode+1:N]),size(X,mode),[]);
end