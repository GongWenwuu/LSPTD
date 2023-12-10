function lip = lipU(U, G, L, T, gamma, mode, flag, delta)
% L_{\mathbf{U}_{n}} =
% \left\|{\mathbf{G}_{(n)}} {\mathbf{V}}^{\mathrm{T}}_{n} {\mathbf{V}}\mathbf{G}_{(n)}^{\mathrm{T}}\right\|_{\mathrm{F}}
% \left\| \alpha \mathbf{L} \right\|_{\mathrm{F}}
Bsq = Matrixization(G,U,mode,'decompress')*Unfold(G,mode)';
if flag == 1
    lip = norm(Bsq,2) + gamma*norm(L,2);
elseif flag == 0
    lip = norm(Bsq,2) + gamma*norm((T*T'),2);
else
    lip = norm(Bsq,2);
end
lip = delta*lip;
end

function X_unf = Unfold(X,mode)
N = ndims(X);
X_unf = reshape(permute(X, [mode 1:mode-1 mode+1:N]),size(X,mode),[]);
end