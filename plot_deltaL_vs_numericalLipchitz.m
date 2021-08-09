% Copyright (c) by Jonas Umlauft (TUM) under BSD License 
% Last modified: Jonas Umlauft 2019-10
clearvars; clear; close all; clc;rng default; setname = '2D';

%check Matlab version
vers=version('-release');
if(str2double(vers(end-2:end-1))<17)
    error('MATLAB R2017a or a newer release is required!');
end

%% Set Parameters
disp('Setting Parameters...')

% Basic Parameters
Ntr = 100;    % Number of training points
Tsim = 30;   % Simulation time
Nsim = 200;  % Simulation steps
sn = 0.1;     % Observation noise (std deviation)
E = 2;        % State space dimension

% Initial State /reference for simulation
x0 = [0 0]';
ref = @(t) refGeneral(t,E+1,@(tau) 2*sin(tau));  % circle


% Controller gains
pFeLi.lam = ones(E-1,1);
pFeLi.kc = 2;

% Define Systemdynamics
a = 1; b = 1; c = 0;
pdyn.f = @(x) 1-sin(x(1,:)) + b*sigmf(x(2,:),[a c]);
pdyn.g = @(x) 1;


% GP learning and simulation  parameters
optGPR = {'KernelFunction','ardsquaredexponential','ConstantSigma',true,'Sigma',sn};
odeopt = odeset('RelTol',1e-3,'AbsTol',1e-6);

% Visualization
Nte = 1e4; XteMin = [-6 -4]; XteMax = [4 4]; % Xrange and n data points
Ndte = floor(nthroot(Nte,E));  Nte = Ndte^E; % Round to a power of E
Xte = ndgridj(XteMin, XteMax,Ndte*ones(E,1)) ; % Generate the grid
Xte1 = reshape(Xte(1,:),Ndte,Ndte); Xte2 = reshape(Xte(2,:),Ndte,Ndte); % Separate the dimensions
Ntrajplot = 100;

% Lyapunov test
tau = 1e-8;     % Grid distance
delta = 0.01;     % Probability for error bound
% deltaL = 0.01;     % Probability for Lipschitz constant

%%  Generate Training Points
disp('Generating Training Points...')
Ntr = floor(nthroot(Ntr,E))^E;
Xtr = ndgridj([0 -3],[3 3],sqrt(Ntr)*ones(E,1));
Ytr = pdyn.f(Xtr) +  sn.*randn(1,Ntr);

%% Learn Model - Optimize Hyperparameters
disp('Learning GP model...')
gprModel = fitrgp(Xtr',Ytr',optGPR{:});
pFeLi.f = @(x) predict(gprModel,x'); pFeLi.g = pdyn.g;

sigfun = @(x) nth_output(2, @predict, gprModel,x');
kfcn = gprModel.Impl.Kernel.makeKernelAsFunctionOfXNXM(gprModel.Impl.ThetaHat); % What does this do?
ls = exp(gprModel.Impl.ThetaHat(1:E));  sf = exp(gprModel.Impl.ThetaHat(end));


%% Test Lyapunov condition
negLogDeltaLs = linspace(1, 6, 100);
Lfhs = []; deltaLs = [];
Lfs = []; Lfprobs = [];
for negLogDeltaL = negLogDeltaLs
    deltaL = 10^(-negLogDeltaL);
    deltaLs = [deltaLs; deltaL];
    [Lfh, Lf, Lfprob] = numericalLipschitz(gprModel, pdyn, Xte, deltaL, Nte, E);
    Lfhs = [Lfhs; Lfh];
    Lfs = [Lfs; Lf];
    Lfprobs = [Lfprobs; Lfprob];
end

f1 = figure;
semilogx(deltaLs, Lfhs);
title('\delta_L vs L_f analytical');
xlabel('\delta_L');
ylabel('L_f');
print(f1, 'delta_L_vs_L_f_analytical.pdf', '-dpdf', '-bestfit');
f2 = figure;
semilogx(Lfprobs, Lfs, '*');
title('\delta_L vs L_f numerical');
xlabel('\delta_L');
ylabel('L_f');
print(f2, 'delta_L_vs_L_f_numerical.pdf', '-dpdf', '-bestfit');