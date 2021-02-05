
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% DATA MINING AND NEURAL NETWORKS                                   %%%%
%%%% Prof. Dr. ir. Johan A. K. Suykens                                 %%%%
%%%% Assignment 1 - Part 4                                             %%%%
%%%% The Curse of Dimensionality                                       %%%%
%%%%                                                                   %%%%
%%%% Copyright: HENRI DE PLAEN, KU LEUVEN                              %%%%
%%%% Date: 11 October 2019                                             %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d = 7 ;                % dimension
R = 7 ;              % domain radius

% DATASETS
n_train = 10000 ;                            % training set size
n_test  = 100  ;                            % test set size

s_train = .1 ;                              % noise variance of the training set
s_test  = .0 ;                              % noise variance of the test set


p = 1 ;                % order of the polynomial (the total number of model 
                                            % parameters will be a combination of d+p out of p)


    
% NEURAL NETWORK
n_neurons = [4,2] ;                         % number of neurons per hidden layer


Train_input  = randsphere(n_train, d, R) ;                      % samples training set on the hyper-sphere 
Test_input   = randsphere(n_test,  d, R) ;                      % samples test set on the hyper-sphere

Train_norms  = sqrt(sum(Train_input.^2,2)) ;                    % computes euclidean norm of each datapoint of the training set
Test_norms   = sqrt(sum(Test_input.^2, 2)) ;                    % computes euclidean norm of each datapoint of the test set

Train_output = sinc(Train_norms) ;                              % computes the cardinal sinus of each norm of the training set
Test_output  = sinc(Test_norms) ;                               % computes the cardinal sinus of each norm of the test set

Train_output_noisy = Train_output + s_train*randn(n_train,1) ;  % adds eventual noise to the training set   (won't change anything if s=0)
Test_output_noisy  = Test_output  + s_test *randn(n_test, 1) ;  % adds eventual noise to the test set       (won't change anything if s=0)

%  mat_poly = zeros(8,2);
% n_param_poly = zeros(8,2);

tic ;                                                                           % starts the times for the polynomial
mdl_poly = polyfitn(Train_input,Train_output_noisy,p) ;                         % (training) performs the multi-dimensional polynomial regression on the training set
time_poly = toc ;                                                               % stops the timer and saves the time of training the polynomial

Poly_train_output = polyvaln(mdl_poly,Train_input) ;                            % evaluates the test inputs on the trained polynomial model
rmse_poly_train = 1/n_train*sqrt(sum((Poly_train_output-Train_output).^2)) ;    % computes root mean square error on the test set
Poly_test_output = polyvaln(mdl_poly,Test_input) ;                              % evaluates the test inputs on the trained polynomial model
rmse_poly_test = 1/n_test*sqrt(sum((Poly_test_output-Test_output).^2)) ; 

mdl_nn = feedforwardnet(n_neurons,'trainlm') ;                              % creates the feedforward neural network
mdl_nn.trainParam.showWindow = false ;                                      % avoid plotting output window
tic ;                                                                       % starts the timer for the training of the neural network
mdl_nn = train(mdl_nn,Train_input',Train_output') ;                         % trains the network
time_nn = toc ;                                                             % stops the timer and saves the training time of the neural network

NN_train_output = mdl_nn(Train_input') ;                                    % evaluates the network on the test set
rmse_nn_train = 1/n_test*sqrt(sum((NN_train_output'-Train_output).^2)) ;    % computes root mean square error on the test set
NN_test_output = mdl_nn(Test_input') ;                                      % evaluates the network on the test set
rmse_nn_test = 1/n_test*sqrt(sum((NN_test_output'-Test_output).^2)) ;       % computes root mean square error on the test set


vol           = pi^(d/2)*R^d/gamma(d/2+1) ;                                         % volume of the domain hyper-sphere
n_params_poly = nchoosek(d+p,p) ;                                                   % number of parameters for the polynomial model
n_params_nn   = sum(n_neurons) + sum([d n_neurons 1 0].*[0 d n_neurons 1]) ;        % number of parameters for the neural network model

% PRINT
fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n') ;
fprintf('%%%%  PROBLEM                                                     %%%% \n') ;
fprintf('%%%%  Dimension: %2i                                               %%%% \n',d) ;
fprintf('%%%%  Domain:  radius=%2.1f      volume=%2.2e                    %%%% \n',R,vol) ;
fprintf('%%%%  Training set size: %i                                     %%%% \n',n_train) ;
fprintf('%%%%  Test set size: %i                                          %%%% \n',n_test) ;
fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n') ;
fprintf('%%%%                           POLYNOMIAL        NEURAL NETWORK   %%%% \n') ;
fprintf('%%%%  Number of parameters:    %4i             %3i               %%%% \n', n_params_poly, n_params_nn) ;
fprintf('%%%%  Datapoints/parameter:    %4.1f             %4.1f            %%%% \n', n_train/n_params_poly, n_train/n_params_nn) ;
fprintf('%%%%  RMSE (Train):            %3.2e          %3.2e         %%%% \n', rmse_poly_train, rmse_nn_train) ;
fprintf('%%%%  RMSE (Test):             %3.2e          %3.2e         %%%% \n', rmse_poly_test, rmse_nn_test) ;
fprintf('%%%%  Training time [s]:       %3.2e          %3.2e         %%%% \n', time_poly, time_nn) ;
fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n') ;

v=[vol]
r=[rmse_nn_test,rmse_poly_test]

% volume= zeros(1,2);
% rmsenn= zeros(1,2);
% n_param_nn= zeros(1,2);
n_param_nn(1,2)=n_params_nn;
rmsenn(1,2)=rmse_nn_test; 
volume(1,2)= v;

x=1:1:8;
figure
hold on 
plot(x,mat_poly(:,1))
title('MSE vs polynomial degree for d=2')
xlabel('Degree of Polynomial')
ylabel('MSE')

figure
x=1:1:8;
plot(x,mat_poly(:,2))
title('MSE vs polynomial degree for d=7')
xlabel('Degree of Polynomial')
ylabel('MSE')

