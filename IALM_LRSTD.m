function [Xest, U, hist] = IALM_LRSTD(X,Omega,Opts) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                          Inexact Augmented Lagrange Multiplier Framework for LRSTD                                        %
% \underset{\mathcal{G},\{\mathbf{U}_{n}\}}{\operatorname{min}} \ (1-\alpha) \prod_{n=1}^{N}\left\|\mathbf{U}_{n}\right\|_{*} +\alpha\|\mathcal{G}\|_{1}}
% \frac{\mu}{2} \left\|\mathcal{X}-\mathcal{G} \times_{n=1}^{N} \mathbf{U}_{n}\right\|_{\mathrm{F}}^{2} + \left\langle\mathcal{Y}, \mathcal{X}-\mathcal{G} \times_{n=1}^{N} \mathbf{U}_{n} \right\rangle
%                                       This code was written by Wenwu Gong (2023.03)                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(Opts, 'flag');flag = [];           else; flag = Opts.flag;     end
if ~isfield(Opts, 'mu');  mu = 1e-5;           else; mu = Opts.mu;         end
if isfield(Opts,'maxit'); maxit = Opts.maxit;  else; maxit = 500;   end
if isfield(Opts,'nu');    nu = Opts.nu;        else; nu = 1;        end
if isfield(Opts,'phi');   phi = Opts.phi;      else; phi = 0;       end
if isfield(Opts,'tol');   tol = Opts.tol;      else; tol = 1e-3;    end
if isfield(Opts, 'epsilon'); epsilon = Opts.epsilon; else; epsilon = 1e-3; end 

N = ndims(X);
Z = X.*Omega;
[G_pre,U_pre,~] = Initial(Z,Opts.Rpara,Opts.init);  

mu_rho = 1.15;
sample_ra = sum(Omega(:))/numel(X);
if sample_ra < 0.05
    mu_rho = 1.1;
end
Y = zeros(size(X));

[L, T, gamma, flag] = localprior(X, size(G_pre), flag, 'lrstd');
w_n = myweight(X, N, 'sum', Opts.alpha);

% time = tic; 
% figure('Position',get(0,'ScreenSize'));
niter = 0; U = U_pre;
% t0 = 1; Lg_pre = 1; LU_pre = ones(N,1); 
% LUnew = ones(N,1); wU = ones(N,1); 
for iter = 1:maxit

    [G, ~] = apgG(G_pre,U_pre,Z,Y,1,mu);

    for n = 1:N
        [U{n}, ~] = apgU(U_pre,G,Z,Y,L{n},T{n},w_n(n),gamma(n),flag(n),mu,n);
    end
    
    % objk = loss(X, Omega, G, U, w_n, Opts); 
    % hist.obj(iter) = objk;
    % t = (1+sqrt(1+4*t0^2))/2;
    % w = (t0-1)/t;
    % wG = min([w,0.999*sqrt(Lg_pre/Lgnew)]);
    % G = G + wG*(G - G_pre);
    % Lg_pre = Lgnew;
    % for i = 1:N
    %     wU(i) = min([w,0.9999*sqrt(LU_pre(i)/LUnew(i))]);
    %     U{i} = U{i}+wU(i)*(U{i}-U_pre{i});
    %     LU_pre(n) = LUnew(n);
    % end
    % t0 = t;
    G_pre = G; U_pre = U; 

    Z_pre = Z;
    Z_new = ModalProduct_All(G,U,'decompress');
    Z(~Omega) = Z_new(~Omega) - Y(~Omega)/mu;
    Z(Omega) = X(Omega) + phi*(Z(Omega) - Z_new(Omega));

    Y = Y + mu*nu*(Z - Z_new);
    mu = mu*mu_rho;

    % -- diagnostics and reporting --
    relchange = norm(Z(:)-Z_pre(:))/norm(Z(:));
    hist.rel(1,iter) = relchange;
    relerr = norm(Z(:)-X(:))/TensorNorm(X,'fro');
    hist.rel(2,iter) = relerr;
    hist.rel(3,iter) = TensorNorm(Z-Z_new,'fro')/TensorNorm(Z,'fro');
    rmse = sqrt((1/length(nonzeros(~Omega)))*norm(X(~Omega)-Z(~Omega),2)^2);
    hist.rmse(iter) = rmse;
    hist.rse(iter) = norm(X(~Omega)-Z_new(~Omega))/norm(X(:));
    nmae = norm(X(~Omega)-Z(~Omega),1)/norm(X(~Omega),1);
    hist.nmae(iter) = nmae;
    
    % if mod(iter,10)==0 
    %      disp(['LRSTD completed at ',int2str(iter),'-th iteration step within ',num2str(toc(time)),' seconds ']);
    %      fprintf('===================================\n');
    %      fprintf('Objective = %e\t, rel_DeltaX = %d\t,NMAE = %d\t,RMSE = %d\n',objk, relchange,nmae,rmse);
    % end
    % plot(hist.nmae);title('# iterations vs. NMAEs');
    % pause(0.1);
    
    % -- stopping checks --
    if relerr < epsilon; niter = niter +1; else; niter = 0; end
    if relchange < tol || niter > 2
        break;
    end
    
end

Xest = Z;

end