function [Xest, U, hist] = PALM_LRSTD(X,Omega,Opts) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                       Proximal Alternating Linearized Minimization for E-LRSTD                                    %
% \underset{\mathcal{G},\{\mathbf{U}_{n}\}}{\operatorname{min}} \ (1-\alpha) \prod_{n=1}^{N}\left\|\mathbf{U}_{n}\right\|_{*} + \alpha\|\mathcal{G}\|_{1} 
% + \sum_{n=1}^{N} \frac{\beta_n}{2} \operatorname{tr}\left(\mathbf{U}_{n}^{\mathrm{T}} \mathbf{L}_n \mathbf{U}_{n}\right) + \frac{\beta}{2}\left\|\mathcal{G} \times_{n=1}^{N} \mathbf{U}_{n} -\mathcal{X}^{0}\right\|_{\mathrm{F}}^{2}
%                                       This code was written by Wenwu Gong (2023.04)                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(Opts, 'epsilon'); epsilon = Opts.epsilon; else; epsilon = 1e-3; end 
if isfield(Opts,'maxit');    maxit = Opts.maxit;     else; maxit = 500;    end
if isfield(Opts,'tol');      tol = Opts.tol;         else; tol = 1e-5;     end
if isfield(Opts,'phi');      phi = Opts.phi;         else; phi = 0;        end
if isfield(Opts,'delta');    delta = Opts.delta;     else; delta = 1.1;    end

N = ndims(X);
Z = X.*Omega;
[Ginit,Uinit,~] = Initial(Z,Opts.Rpara,Opts.init); 

gamma = zeros(1,N); L = cell(1,N); T = cell(1,N);
if isfield(Opts, 'flag')
    for n = 1: N
        Xmat = reshape(permute(X, [n 1:n-1 n+1:N]),size(X,n),[]);
        if Opts.flag(n) == 1
            if strcmp(Opts.prior,'stdc')
                L = constructL_stdc(size(X), {1,2,3}, 5, L);
            elseif strcmp(Opts.prior,'lrstd') || strcmp(Opts.init,'rtd')
                L = constructL_lrstd(X,5,0);
            elseif strcmp(Opts.prior,'l2')
                L{n} = eye(size(X,n));
            elseif strcmp(Opts.prior, 'toep')
                L{n} = constructT(size(X, n));
                L{n} = L{n}*L{n}';
            end
            gamma(n) = norm(Xmat,2)/(2*norm(L{n},2));
        elseif Opts.flag(n) == 0
            if strcmp(Opts.prior,'l2')
                T{n} = eye(size(Ginit,n));
                gamma(n) = norm(Xmat,2)/N;
            else
                T{n} = constructT(size(Ginit,n));
                gamma(n) = norm(Xmat,2)/(2*norm(T{n}*T{n}',2));
            end
        end
    end
else
    Opts.flag = 2.*ones(1,N);
end

Usq = cell(1,N); w_n = ones(1,N);
for n = 1:N
    Usq{n} = Uinit{n}'*Uinit{n};
    if strcmp(Opts.init,'lrstd')
        if ~isfield(Opts, 'weight')
            w_n(n) = Weight(X, n, Opts.alpha,[]);
        elseif isfield(Opts, 'weight')
            if strcmp(Opts.weight,'sum')
                w_n(n) = Weight(X, n, Opts.alpha, Opts.weight);
            end
        end
    end
end

obj0 = loss(X, Omega, Ginit, Uinit, w_n, Opts);

t0 = 1; niter = 0;
Gextra = Ginit; Uextra = Uinit; U = Uinit; 
Lgnew = 1; LU0 = ones(N,1); LUnew = ones(N,1);
gradU = cell(N,1); wU = ones(N,1);

% time = tic;
% figure('Position',get(0,'ScreenSize'));
for iter = 1:maxit
    
    for n = 1:N
        % -- Core tensor updating --    
        gradG = gradientG(Gextra, U, Usq, Z); 
        Lg0 = Lgnew;
        Lgnew = lipG(Usq, delta);
        if strcmp(Opts.init,'sntd')
            G = max(0,abs(Gextra - gradG/Lgnew) - Opts.alpha/(Opts.beta*Lgnew));
        elseif strcmp(Opts.init,'rtd')
            G = thresholding(Gextra - gradG/Lgnew, Opts.alpha/(Opts.beta*Lgnew));
        elseif strcmp(Opts.init,'lrstd')
            if Opts.alpha == 0
                G = Gextra - gradG/Lgnew;
            else
                G = thresholding(Gextra - gradG/Lgnew, 1/(Opts.beta*Lgnew));
            end
        end

        % -- Factor matrices updating --
        gradU{n} = gradientU(Uextra, U, Usq, G, Z, L{n}, T{n}, gamma(n), n, Opts.flag(n));
        LU0(n) = LUnew(n);
        LUnew(n) = lipU(Usq, G, L{n}, T{n}, gamma(n), n, Opts.flag(n), delta);
        if strcmp(Opts.init,'rtd')
            U{n} = max(0,Uextra{n} - gradU{n}/(Opts.beta*LUnew(n)));
        elseif strcmp(Opts.init,'sntd')
            U{n} = max(0,Uextra{n} - gradU{n}/LUnew(n) - Opts.alpha/(Opts.beta*LUnew(n)));
        elseif strcmp(Opts.init,'lrstd')
            if isfield(Opts, 'weight')
                if strcmp(Opts.weight,'prod') || strcmp(Opts.weight,'lrstd')
                    w_n(n) = Weight(Uextra, n, Opts.alpha, Opts.weight);
                end
            end
            if Opts.alpha == 1
                U{n} = Uextra{n} - gradU{n}/LUnew(n);
            else
                [U{n}, ~, ~] = tracenorm(Uextra{n} - gradU{n}/LUnew(n), w_n(n)/(Opts.beta*LUnew(n)));
            end
        end

        Usq{n} = U{n}'*U{n};
    end

    if strcmp(Opts.init,'lrstd')
        Gextra = G;
        gradG = gradientG(Gextra, U, Usq, Z); 
        Lg0 = Lgnew;
        Lgnew = lipG(Usq, delta);
        if Opts.alpha == 0
            G = Gextra - gradG/Lgnew;
        else
            G = thresholding(Gextra - gradG/Lgnew, 1/(Opts.beta*Lgnew));
        end
    end
   
    Z_pre = Z;
    Z_new = ModalProduct_All(G,U,'decompress');
    Z(~Omega) = Z_new(~Omega);
    Z(Omega) = X(Omega) + phi*(Z(Omega) - Z_new(Omega));

    % -- diagnostics and reporting --
    objk = loss(X, Omega, G, U, w_n, Opts); 
    hist.obj(iter) = objk;
    relchange = norm(Z(:)-Z_pre(:))/norm(Z(:));
    hist.rel(1,iter) = relchange;
    relerr = norm(X(Omega)-Z_new(Omega),2)/norm(Z_new(Omega),2);
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
    %      fprintf('Objective = %e\t, rel_DeltaX = %d\t,nmae = %d\t,RMSE = %d\n',objk, relchange,nmae,rmse);
    % end
    % plot(hist.nmae);title('# iterations vs. NMAEs');
    % pause(0.1);
    % -- stopping checks and correction -- 
    if relerr < epsilon; niter = niter +1; else; niter = 0; end
    if relchange < tol || niter > 2
        break;
    end 
    
    % -- extrapolation --      
    t = (1+sqrt(1+4*t0^2))/2;
    if objk >= obj0
        Gextra = Ginit;
        Uextra = Uinit;
    else 
        w = (t0-1)/t;
        wG = min([w,0.999*sqrt(Lg0/Lgnew)]);
        Gextra = G + wG*(G - Ginit); 
        for n = 1:N
            wU(n) = min([w,0.9999*sqrt(LU0(n)/LUnew(n))]);
            Uextra{n} = U{n}+wU(n)*(U{n}-Uinit{n});
        end
        Ginit = G; Uinit = U; t0 = t; obj0 = objk;
    end
    
end

Xest = Z_new;
Xest(Omega) = X(Omega);

end