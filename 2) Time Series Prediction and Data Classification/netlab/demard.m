load("breast.mat");

% Generate the data set.
x = trainset;
t = labels_train;
t(t==-1)=0;

randn('state', 0); 
rand('state', 0);

% ndata = 100;
% 
% x = [x1, x2, x3];noise = 0.05;
% x1 = rand(ndata, 1) + 0.002*randn(ndata, 1);
% x2 = x1 + 0.02*randn(ndata, 1);
% x3 = 0.5 + 0.2*randn(ndata, 1);
% t = sin(2*pi*x1) + noise*randn(ndata, 1);

% Set up network parameters.
nin = 30;			% Number of inputs.
nhidden = 2;			% Number of hidden units.
nout = 1;			% Number of outputs.
aw1 = 0.01*ones(1, nin);	% First-layer ARD hyperparameters.
ab1 = 0.01;			% Hyperparameter for hidden unit biases.
aw2 = 0.01;			% Hyperparameter for second-layer weights.
ab2 = 0.01;			% Hyperparameter for output unit biases.
beta = 50.0;			% Coefficient of data error.

% Create and initialize network.
prior = mlpprior(nin, nhidden, nout, aw1, ab1, aw2, ab2);
net = mlp(nin, nhidden, nout, 'linear', prior, beta);

% Set up vector of options for the optimiser.
nouter = 2;			% Number of outer loops
ninner = 10;		        % Number of inner loops
options = zeros(1,18);		% Default options vector.
options(1) = 1;			% This provides display of error values.
options(2) = 1.0e-7;	% This ensures that convergence must occur
options(3) = 1.0e-7;
options(14) = 300;		% Number of training cycles in inner loop. 

% Train using scaled conjugate gradients, re-estimating alpha and beta.
for k = 1:nouter
  net = netopt(net, options, x, t, 'scg');
  [net, gamma] = evidence(net, x, t, ninner);
  fprintf(1, '\n\nRe-estimation cycle %d:\n', k);
  disp('The first alphas are the hyperparameters for the corresponding');
  disp('input to hidden unit weights.  The remainder are the hyperparameters');
  disp('for the hidden unit biases, second layer weights and output unit')
  disp('biases, respectively.')
  fprintf(1, '  alpha =  %8.5f\n', net.alpha);
  fprintf(1, '  beta  =  %8.5f\n', net.beta);
  fprintf(1, '  gamma =  %8.5f\n\n', gamma);

end



disp('We can now read off the hyperparameter values corresponding to the')
disp('inputs :')
disp(' ');
fprintf(1, '    alpha1: %8.5f\n', net.alpha(1));
fprintf(1, '    alpha2: %8.5f\n', net.alpha(2));
fprintf(1, '    alpha3: %8.5f\n', net.alpha(3));
disp(' ');

disp('weight values:')
disp(' ');
fprintf(1, '    %8.5f    %8.5f\n', net.w1');
disp(' ');
disp('where the three rows correspond to weights asssociated with x1, x2 and')
disp('x3 respectively. We see that the network is giving greatest emphasis')
disp('to x1 and least emphasis to x3, with intermediate emphasis on')
disp('x2. Since the target t is statistically independent of x3 we might')
disp('expect the weights associated with this input would go to')
disp('zero. However, for any finite data set there may be some chance')
disp('correlation between x3 and t, and so the corresponding alpha remains')
disp('finite.')

disp(' ');
disp('Press any key to end.')
%pause; clc; close(h); clear all
alphaval = net.alpha;
alphaval= alphaval([1:30],:);
[sortval,sortidx]=sort(alphaval,'ascend');