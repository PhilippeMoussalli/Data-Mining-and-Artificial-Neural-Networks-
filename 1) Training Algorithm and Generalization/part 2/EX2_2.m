%% Ex 1.2
%Student number r0778043

load('data_personal_regression_problem.mat')
Tnew = (8*T1+7*T2+7*T3+4*T4+3*T5)/(8+7+7+4+3);
X=[X1';X2';Tnew'];
Net_Data = datasample(X,3000,2,'Replace',false); % Draw random samples 


X_train = Net_Data(1:2,:) ;  
T_train = Net_Data(3,:) ;    
X1 = Net_Data(1,:);
X2 = Net_Data(2,:);


net1 = feedforwardnet(10,'trainlm') ;   % creates a forst net with the 'trainlm' algorithm
net1.inputs{1}.size = 2; 

% Define training, validation and test set (1000 samples each)

net1.divideFcn= 'dividerand'; % divide the data randomly 
net1.divideParam.trainRatio= 0.33; 
net1.divideParam.valRatio= 0.33; 
net1.divideParam.testRatio= 0.33; 
net1.layers{1}.transferFcn = 'tansig';

view(net1)

[net1,tr1] = train(net1,X_train,T_train) ;

network_output = net1(X_train);

trOut = network_output(tr1.trainInd);
vOut = network_output(tr1.valInd);
tsOut = network_output(tr1.testInd); %%
trTarg = T_train(tr1.trainInd);
vTarg = T_train(tr1.valInd);
tsTarg = T_train(tr1.testInd);  %%


% figure
% plotregression(trTarg, trOut, 'Train', vTarg, vOut, 'Validation', tsTarg, tsOut, 'Testing')
% 
% plotperform(tr1)

% Extract tested values for evaluation (Compare testset with network
% output)
X1_T = X1(tr1.testInd);
X2_T = X2(tr1.testInd);
T_T = T_train (tr1.testInd);  %Original test set value 

[xq,yq] = meshgrid(-1:.07:1, -1:.07:1);

%Interpolation to creat surface plot 
vq1 = griddata(X1_T,X2_T,tsTarg,xq,yq);
vq2 = griddata(X1_T,X2_T,tsOut,xq,yq);
%% 6 neurons hidden layer
vq2 = griddata(X1_T,X2_T,tsOut,xq,yq);
%% 2 neurons in hidden layer
vq3 = griddata(X1_T,X2_T,tsOut,xq,yq);
%% 10 neurons in hidden layer 
vq4 = griddata(X1_T,X2_T,tsOut,xq,yq);



%Plotting of surface for both test set and network output

figure
s1= mesh(xq,yq,vq1); 
s1.FaceColor = 'flat';
colorbar
xlabel('X1')
ylabel('X2')
zlabel('T')
zlim([2 4])
title('Surface plot of test set')


figure
s2= mesh(xq,yq,vq2);
s2.FaceColor = 'flat';
colorbar
xlabel('X1')
ylabel('X2')
zlabel('T')
zlim([2 4])
title('Surface plot of output set')

figure
hold on 
contour(xq,yq,vq1,10,'LineColor','r')
contour(xq,yq,vq2,10,'LineColor','b')
contour(xq,yq,vq3,10,'LineColor','g')
contour(xq,yq,vq4,10,'LineColor','k')
xlim([0 1]) 
ylim([0 1])
legend('Test set','2 layers','6 layers','10 layers')


perf = mse(net1,tsTarg,tsOut)