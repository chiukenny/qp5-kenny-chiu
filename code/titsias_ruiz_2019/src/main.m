function main(model_name, data_name, method_name, BurnIters, SamplingIters)
% This is the code for UIVI.
% 
% INPUTS:
%   + model_name: Specify the model. It can take on one of these values:
%     {'banana', 'gaussmix'}
%   + data_name: Indicate the dataset. It can take on one of these values:
%      - For gaussmix: {'xshape', 'multimodal'}
%      - For 'banana': { [empty] }
%   + method_name: Specify the method. It can take on one of these values:
%     {'sivi', 'usivi'}
%   + BurnIters: Number of burning iterations for the HMC chain
%     [default: 5]
%   + SamplingIters: Number of samples obtaine in the HMC procedure
%     [default: 5]
%
%% 

% Optional inputs
if(nargin<=3)
    BurnIters = 5;
end
if(nargin<=4)
    SamplingIters = 5;
end
namelabel = ['_burnIters' num2str(BurnIters) '_samplingIters' num2str(SamplingIters)];

% Add path
addpath nnet/;
addpath mcmc/;
addpath auxil/;
addpath approx/;
addpath logdensities/;
addpath plots/;

% Parameters
param.outdir = '../out/';               % Output directory
pxz.model = model_name;
pxz.data = data_name;
param.Nsamples = 1;
param.method = method_name;

% Random seed
randn('seed',1);
rand('seed',1);

%% Model and data
param.flag_minibatch = 0;
if(strcmp(pxz.model, 'banana'))
    % Parameters
    param.iters = 50000;            % Number of iterations
    param.dim_noise = 1;            % Dimensionality of epsilon
    
    % Model definition
    dim_z = 1;                      % Dimensionality of z
    pxz.logdensity = @logdensityBanana2D;
    pxz.inargs{1} = [0 0];          % Mean vector data
    Sigma = [1 0.9; 0.9 1];
    pxz.inargs{2} = chol(Sigma)';   % Cholesky decomposition of the covariance matrix
    pxz.inargs{3} = 1;              % 1st bananity parameter
    pxz.inargs{4} = 1;              % 2nd bananity parameter
    % Model name
    pxz.dataName = 'banana';
elseif(strcmp(pxz.model, 'banana3D'))
    % Parameters
    param.iters = 50000;            % Number of iterations
    param.dim_noise = 1;            % Dimensionality of epsilon
    
    % Model definition
    dim_z = 3;                      % Dimensionality of z
    pxz.logdensity = @logdensityBanana3D;
    pxz.inargs{1} = [0 0 0];          % Mean vector data
    Sigma = [1 0.9 0.9; 0.9 1 0.9; 0.9 0.9 1];
    pxz.inargs{2} = chol(Sigma)';   % Cholesky decomposition of the covariance matrix
    pxz.inargs{3} = 1;              % 1st bananity parameter
    pxz.inargs{4} = 1;              % 2nd bananity parameter
    % Model name
    pxz.dataName = 'banana3D';
elseif(strcmp(pxz.model, 'bananaND'))
    % Parameters
    param.iters = 50000;            % Number of iterations
    param.dim_noise = 1;            % Dimensionality of epsilon
    
    % Model definition
    dim_z = 50;                      % Dimensionality of z
    pxz.logdensity = @logdensityBananaND;
    pxz.inargs{1} = zeros(1,dim_z);          % Mean vector data
    Sigma = ones(dim_z)*0.9 + diag(ones(dim_z,1))*0.1;
    pxz.inargs{2} = chol(Sigma)';   % Cholesky decomposition of the covariance matrix
    pxz.inargs{3} = 1;              % 1st bananity parameter
    pxz.inargs{4} = 1;              % 2nd bananity parameter
    % Model name
    pxz.dataName = 'bananaND';
elseif(strcmp(pxz.model, 'gaussmix'))
    % Parameters
    param.iters = 50000;            % Number of iterations
    param.dim_noise = 3;           % Dimensionality of epsilon
    % Model definition
    dim_z = 2;                      % Dimensionality of z
    if(strcmp(pxz.data, 'xshape'))
        weightsMixture = [0.5 0.5];
        muGaussians = {[0,0], [0,0]};   % Mean of each Gaussian component
        SigmaGaussians = {[2 1.8; 1.8 2], [2 -1.8; -1.8 2]}; % Covariance of each Gaussian component
    elseif(strcmp(pxz.data, 'multimodal'))
        weightsMixture = [0.5 0.5];
        muGaussians = {[-2,0], [2,0]};     % Mean of each Gaussian component
        SigmaGaussians = {eye(2), eye(2)}; % Covariance of each Gaussian component
    else
        error(['Unknown data name: ' pxz.data]);
    end
    L = cell(1, length(SigmaGaussians));   % Cholesky decomposition of the covariance matrices
    for cc=1:length(SigmaGaussians)
        L{cc} = chol(SigmaGaussians{cc})';
    end
    pxz.logdensity = @logdensityGaussianMixture; 
    pxz.inargs{1} = muGaussians;  % mean vectors
    pxz.inargs{2} = L;            % Cholesky decomp of the covariances
    pxz.inargs{3} = weightsMixture;   % mixture weights
    % Model name
    pxz.dataName = ['gaussmix_' pxz.data];
else
    error(['Model not known: ' pxz.model]);
end

%% Define the implicit variational distribution

% Neural network
if(strcmp(pxz.model, 'banana') || strcmp(pxz.model, 'banana3D') || strcmp(pxz.model, 'bananaND'))
    param.nn.numUnitsPerHiddenLayer = [50 50];    % units per hidden layer (the length of this vector is the number of hidden layers)
    param.nn.numUnitsPerLayer = [dim_z param.nn.numUnitsPerHiddenLayer param.dim_noise];  % all units from output (left) to the input (right)
    param.nn.act_funcs = {'lin', 'relu', 'relu'};    % activations functions {'relu' 'softmax' 'lin' 'cos' 'sigmoid' 'tanh' 'softplus'}
elseif(strcmp(pxz.model, 'gaussmix'))
    param.nn.numUnitsPerHiddenLayer = [50 50];    % units per hidden layer (the length of this vector is the number of hidden layers)
    param.nn.numUnitsPerLayer = [dim_z param.nn.numUnitsPerHiddenLayer param.dim_noise];  % all units from output (left) to the input (right)
    param.nn.act_funcs = {'lin', 'relu', 'relu'};    % activations functions {'relu' 'softmax' 'lin' 'cos' 'sigmoid' 'tanh' 'softplus'}
else
    error(['Unknown model: ' pxz.model]);
end

vardist.net = netcreate(param.nn.numUnitsPerLayer, param.nn.act_funcs);
vardist.sigma = 0.5 * ones(1, dim_z);

% Distribution q(epsilon)
vardist.peps.dim_noise = param.dim_noise; 
vardist.peps.pdf = 'standard_normal';    % 'uniform' not implemented

% Reparameterized conditional
qzEpsilon.logdensity = @logdensity_qzepsilon;
qzEpsilon.inargs{1} = zeros(1, dim_z); 
qzEpsilon.inargs{2} = vardist;

% Parameters for SIVI
if(strcmp(pxz.model, 'banana') || strcmp(pxz.model, 'banana3D') || strcmp(pxz.model, 'bananaND'))
    param.sivi.K = 50;      % parameter for the surrogate ELBO
elseif(strcmp(pxz.model, 'gaussmix'))
    param.sivi.K = 50;      % parameter for the surrogate ELBO
else
    error(['Model not known: ' pxz.model]);
end

% Parameters for the MCMC
param.mcmc.BurnIters = BurnIters;
param.mcmc.SamplingIters = SamplingIters;
param.mcmc.AdaptDuringBurn = 1;
param.mcmc.LF = 5;                % leap frog steps
mcmc.algorithm = @hmc;            % @metropolisHastings; @mala;  
mcmc.inargs{1} = 0.2; % 1/dim_z;  % initial step size parameter delta
mcmc.inargs{2} = param.mcmc.BurnIters;
mcmc.inargs{3} = param.mcmc.SamplingIters; 
mcmc.inargs{4} = param.mcmc.AdaptDuringBurn; 
mcmc.inargs{5} = param.mcmc.LF;

%% Optimization parameters
if(strcmp(pxz.model, 'banana') || strcmp(pxz.model, 'banana3D') || strcmp(pxz.model, 'bananaND') || strcmp(pxz.model, 'gaussmix'))
    param.optim.rhotheta = 0.01;
    param.optim.rhosigma = 0.002;
    param.optim.ReducedBy = 0.9; 
    param.optim.ReducedEvery = 3000;
else
    error(['Model not known: ' pxz.model]);
end
param.optim.kappa0 = 0.1;
param.optim.tau = 1;

%% Initialize Gt (adaptive stepsize parameters)
Gt.net = cell(1, length(vardist.net));
for layer=1:length(vardist.net)-1
    Gt.net{layer}.W = zeros(size( vardist.net{layer}.W ));
    Gt.net{layer}.b = zeros(size( vardist.net{layer}.b ));
end
Gt.sigma = zeros(size( vardist.sigma )); 

%% VI Algorithm

% Stochastic bound at each iteration
out.stochasticBound = zeros(1, param.iters);
out.elbo = zeros(1, param.iters);
out.llh_test = zeros(1, param.iters);
% Average acceptance history and rate for all MCMC chains
out.acceptHist = zeros(1, param.mcmc.BurnIters+param.mcmc.SamplingIters);
out.acceptRate = 0;
% Minibatch state
if(param.flag_minibatch)
    data.batch.st = 1; 
    data.batch.perm = randperm(data.N);
end
% Time per iteration
out.telapsed = zeros(1, param.iters);
% Algorithm
for it=1:param.iters 
%
    % Start timer
    t_start = tic;

    % Initialize all gradients to 0
    logp = 0;
    grad_theta_W = cell(1, length(vardist.net)-1);
    grad_theta_b = cell(1, length(vardist.net)-1);
    grad_sigma = zeros(size(vardist.sigma));
    for cc=1:length(vardist.net)-1
        grad_theta_W{cc} = zeros(size(vardist.net{cc}.W));
        grad_theta_b{cc} = zeros(size(vardist.net{cc}.b));
    end
    
    % Sample auxiliary noise epsilon_0 for the sivi method and pass it through the NN to obtain the parameters of the conditional
    if(strcmp(param.method, 'sivi'))
        epsilon_0 = randn(param.sivi.K, param.dim_noise);
        net_0 = netforward(vardist.net, epsilon_0);
        Tr_epsilon_0 = net_0{1}.Z;
    end

    % For each Monte Carlo sample
    for ss=1:param.Nsamples
    %
        % Sample the noise epsilon
        nRows = 1;
        
        if strcmp(vardist.peps.pdf,'standard_normal')
            epsilon = randn(nRows, vardist.peps.dim_noise);
        elseif strcmp(vardist.peps.pdf,'uniform')
            epsilon = rand(nRows, vardist.peps.dim_noise);
        end
    
        % Compute z = T(epsilon; theta) + sigma*eta;
        eta = randn(nRows, dim_z); 
        net = netforward(vardist.net, epsilon);
        Tr_epsilon = net{1}.Z;
        z = Tr_epsilon + bsxfun(@times, vardist.sigma, eta);

        if(strcmp(param.method, 'usivi'))
            % Sample from the reverse conditional (MCMC to obtain epsilon_t)
            qzEpsilon.inargs{1} = z; 
            qzEpsilon.inargs{2} = vardist;
            [epsilon_t, samples, extraOutputs] = mcmc.algorithm(epsilon, qzEpsilon, mcmc.inargs{:});
            % Keep track of sampling acceptance rate
            out.acceptHist = out.acceptHist + extraOutputs.acceptHist/(param.iters*param.Nsamples);
            out.acceptRate = out.acceptRate + extraOutputs.accRate/(param.iters*param.Nsamples);
            % In case you adapt the stepsize
            mcmc.inargs{1} = extraOutputs.delta;
            % Take the average across MCMC samples
            Tr_epsilon_t = zeros(nRows, dim_z);
            for s=1:param.mcmc.SamplingIters
                if(length(size(samples))==3)
                    epsilon_t = samples(:,:,s);
                elseif(length(size(samples))==2)
                    epsilon_t = samples(s,:);
                end
                net2 = netforward(vardist.net, epsilon_t);
                Tr_epsilon_t = Tr_epsilon_t + net2{1}.Z / param.mcmc.SamplingIters;
            end
            
            % Evaluate the stochastic gradients
            [logpxz, gradz] = pxz.logdensity(z, pxz.inargs{:});
            
            % Average the log-joint
            logp = logp + logpxz/param.Nsamples;
            
            % Model component + Entropy component
            precond_grad  = gradz + bsxfun(@rdivide, z - Tr_epsilon_t, vardist.sigma.^2);
            [gradW, gradb] = netbackpropagation(net, precond_grad, 1);
            gradS = sum(precond_grad.*eta, 1);

        elseif(strcmp(param.method, 'sivi'))
            % Evaluate the weights
            aux_std_z = bsxfun(@rdivide, bsxfun(@minus, z, [Tr_epsilon; Tr_epsilon_0]), vardist.sigma);
            log_q_k = -0.5*sum(aux_std_z.^2, 2);
            weights_k = softmax(log_q_k, 1);
            
            % Evaluate the stochastic gradients
            [logpxz, gradz] = pxz.logdensity(z, pxz.inargs{:});
            precond_grad = gradz + sum(bsxfun(@times, weights_k, bsxfun(@rdivide, aux_std_z, vardist.sigma)), 1);
            logp = logp + logpxz/param.Nsamples;
            [gradW, gradb] = netbackpropagation(net, precond_grad, 1);
            gradS = sum(precond_grad.*eta, 1);
            
            % Add the gradient of the 2nd term of the entropy
            aux_grad_mean = bsxfun(@rdivide, aux_std_z, vardist.sigma);
            [gradW_2, gradb_2] = netbackpropagation(net, weights_k(1)*aux_grad_mean(1,:), 1);
            [gradW_0, gradb_0] = netbackpropagation(net_0, bsxfun(@times, weights_k(2:end), aux_grad_mean(2:end,:)), 1);
            for cc=1:length(vardist.net)-1
                gradW{cc} = gradW{cc} - gradW_2{cc} - gradW_0{cc};
                gradb{cc} = gradb{cc} - gradb_2{cc} - gradb_0{cc};
            end
            
            % Gradient w.r.t. sigma
            aux_grad_sigma = bsxfun(@plus, -1./vardist.sigma, bsxfun(@rdivide, aux_std_z.^2, vardist.sigma));
            gradS = gradS - sum(bsxfun(@times, weights_k, aux_grad_sigma), 1);
            
        else
            error(['Unknown method: ' param.method]);
        end        

        % Average the gradients across Monte Carlo samples
        for cc=1:length(vardist.net)-1
            grad_theta_W{cc} = grad_theta_W{cc} + gradW{cc}/param.Nsamples;
            grad_theta_b{cc} = grad_theta_b{cc} + gradb{cc}/param.Nsamples;
        end
        grad_sigma = grad_sigma + gradS/param.Nsamples;
    %
    end
    
    % RMSprop update of the parameters
    kappa = param.optim.kappa0;
    if(it==1)
        kappa = 1;
    end
    
    for layer=length(vardist.net)-1:-1:1
        Gt.net{layer}.W = kappa*(grad_theta_W{layer}.^2) + (1-kappa)*Gt.net{layer}.W;
        Gt.net{layer}.b = kappa*(grad_theta_b{layer}.^2) + (1-kappa)*Gt.net{layer}.b;
        vardist.net{layer}.W = vardist.net{layer}.W + param.optim.rhotheta * grad_theta_W{layer} ./ (param.optim.tau+sqrt(  Gt.net{layer}.W ));
        vardist.net{layer}.b = vardist.net{layer}.b + param.optim.rhotheta * grad_theta_b{layer} ./ (param.optim.tau+sqrt(  Gt.net{layer}.b ));    
    end
    Gt.sigma = kappa*(grad_sigma.^2) + (1-kappa)*Gt.sigma;
    vardist.sigma = vardist.sigma + param.optim.rhosigma * grad_sigma ./ (param.optim.tau+sqrt(  Gt.sigma ));
    vardist.sigma(vardist.sigma<0.00001) = 0.00001; % for numerical stability 
       
    % Decrease the stepsize
    if( mod(it, param.optim.ReducedEvery) == 0 )
        param.optim.rhosigma = param.optim.rhosigma * param.optim.ReducedBy;
        param.optim.rhotheta = param.optim.rhotheta * param.optim.ReducedBy;
    end
    
    % Compute elapsed time
    out.telapsed(it) = toc(t_start);
    
    % Compute the ELBO without the entropy term (which is not tractable)
    out.stochasticBound(it) = mean(logp);
    if mod(it,1000) == 0    
        fprintf('Iter=%d, Bound=%f\n', it, out.stochasticBound(it));
    end
    
    % Compute test log-likelihood
    if(mod(it,100)==0)
        out.elbo(it) = compute_elbo(10000, 100, pxz, vardist);
    end
%   
end


%% Make plots

% Plot smoothed ELBO
figure;
smth = 50;
smoothed_stochasticBound = tsmovavg(out.stochasticBound, 's', smth, 2);
%plot(cumsum(out.telapsed), smoothed_stochasticBound, 'r', 'linewidth', 0.5);
plot(linspace(0,param.iters,length(smoothed_stochasticBound)), smoothed_stochasticBound, 'r', 'linewidth', 0.25);
ylim([min(smoothed_stochasticBound(250:end))-1,0]);
title('Smoothed ELBO')
name = [param.outdir pxz.dataName '_' param.method '_ELBO_movavg' num2str(smth)];
figurepdf(9, 5);
print('-dpdf', [name '.pdf']);

% Other plots
name = [param.outdir pxz.model '_' pxz.dataName '_' param.method '_results' namelabel '.mat'];
if(strcmp(pxz.model, 'banana') || strcmp(pxz.model, 'banana3D') || strcmp(pxz.model, 'gaussmix'))
    T = 1000;
    plot_toy;
elseif(strcmp(pxz.model, 'bananaND'))
    % No plot
else
    error(['Unknown model: ' pxz.model]);
end
