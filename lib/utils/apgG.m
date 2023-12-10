function [G, Lip] = apgG(G,U,X,Y,alpha,mu)
% \hat{\mathcal{G}} = \underset{\mathcal{G}}{\operatorname{argmin}}\left\langle\nabla_{\mathcal{G}} f(\tilde{\mathcal{G}}), \mathcal{G}-\tilde{\mathcal{G}}\right\rangle+\frac{L_\mathcal{G}}{2}\|\mathcal{G}-\tilde{\mathcal{G}}\|_{F}^{2}+\frac{1}{\beta}\|\mathcal{G}\|_{1} \\
% = \frac{L_{\mathcal{G}}}{2}\left\|\mathcal{G}-(\tilde{\mathcal{G}}-\frac{1}{L_{\mathcal{G}}} \nabla_{\mathcal{G}} f\left(\tilde{\mathcal{G}}\right))\right\|_{\mathrm{F}}^{2}  + \frac{1}{\beta}\|\mathcal{G}\|_{1}
Usq = U; 
for n = 1:ndims(X)
    Usq{n} = U{n}'*U{n};
end
grad = gradientG(G,U,Usq,X+Y/mu);
Lip = lipG(Usq, 1);
G = Proximal(G,grad,Lip,alpha/mu,'l1');
end