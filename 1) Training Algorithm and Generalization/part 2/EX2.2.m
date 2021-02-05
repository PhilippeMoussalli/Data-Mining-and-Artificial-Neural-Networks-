%% Ex 1.2
%Student number r0778043
load('data_personal_regression_problem.mat')
Tnew = (8*T1+7*T2+7*T3+4*T4+3*T5)/(8+7+7+4+3);
figure
plot(X2+X1,Tnew)