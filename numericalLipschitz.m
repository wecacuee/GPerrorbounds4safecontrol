function [Lfh, Lf, Lfprob] = numericalLipschitz(gprModel, pdyn, Xte, deltaL, Nte, E)

pFeLi.f = @(x) sampleGPR(gprModel,x'); pFeLi.g = pdyn.g;

kfcn = gprModel.Impl.Kernel.makeKernelAsFunctionOfXNXM(gprModel.Impl.ThetaHat); % What does this do?
ls = exp(gprModel.Impl.ThetaHat(1:E));  sf = exp(gprModel.Impl.ThetaHat(end));


%% Test Lyapunov condition
disp('Setup Lyapunov Stability Test...')


Lk = norm(sf^2*exp(-0.5)./ls);

k = @(x,xp) sf^2 * exp(-0.5*sum((x-xp).^2./ls.^2,1));
dkdxi = @(x,xp,i)  -(x(i,:)-xp(i,:))./ls(i)^2 .* k(x,xp);
ddkdxidxpi = @(x,xp,i) ls(i)^(-2) * k(x,xp) +  (x(i,:)-xp(i,:))/ls(i)^2 .*dkdxi(x,xp,i);
dddkdxidxpi = @(x,xp,i) -ls(i)^(-2) * dkdxi(x,xp,i) - ls(i)^(-2) .*dkdxi(x,xp,i) ...
    +  (x(i,:)-xp(i,:))/ls(i)^2 .*ddkdxidxpi(x,xp,i);

%% Compute Lipschitz constant numerically
% 
%%% Compute Lipschitz constant by using derivative kernel 
%%% Could not use it as a function because ddkdxidxpi wont pass as an
%%% argument
grad_f_mu = zeros(size(Xte)); % must be ExN
[~, N] = size(Xte);
grad_f_sigma = zeros(E, N);
for e = 1:E
    grad_f_sigma(e, :) = ddkdxidxpi(Xte, Xte, e);
end
grad_fs_i = randn(E, N).*grad_f_sigma + grad_f_mu;
grad_fs_prob = normpdf(grad_fs_i, grad_f_mu, grad_f_sigma) * 1e-2;
gradnorms = sqrt(sum(grad_fs_i.^2,1));
[Lf, idx] =  max(gradnorms);
Lfprob = grad_fs_prob(idx);
%%% sampleGPRFromKnl pasted

r = max(pdist(Xte')); Lfs = zeros(E,1);
for e=1:E
    maxk = max(ddkdxidxpi(Xte,Xte,e));
    Lkds = zeros(Nte,1);
    for nte = 1:Nte
       Lkds(nte) = max(dddkdxidxpi(Xte,Xte(:,nte),e));
    end
    Lkd = max(Lkds);  
    deltaLdependentTerm = sqrt(2*log(2*E/deltaL))*maxk ; % Eq (11) from the paper
    otherTerm = 12*sqrt(6*E)*max(maxk,sqrt(r*Lkd)); % Eq (11) from the paper
    disp("deltaL dependent term"); disp(deltaLdependentTerm);
    disp("other term"); disp(otherTerm);
    Lfs(e) = deltaLdependentTerm + otherTerm;
end
Lfh =  norm(Lfs); % Eq (11) continued
disp("deltaL"); disp(deltaL);
disp("Computed Lipschtz constant is ");
disp(Lfh); 
disp("Numerical est"); disp(Lf);
disp("Probability of Lf"); disp(Lfprob);
disp("Lh");
end