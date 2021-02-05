%% perform unsupervised learning with SOM  

% Marco Signoretto, March 2011

close all;
clear all;
clc;
% first we generate data uniformely distributed within two
% concentric cylinders

X=2*(rand(5000,3)-.5);
indx=(X(:,1).^2+X(:,2).^2<.6)&(X(:,1).^2+X(:,2).^2>.1);
X=X(indx,:)';

% we then initialize the SOM with hextop as topology function
% ,linkdist as distance function and gridsize 5x5x5
net = newsom(X,[5 5 5],'hextop','linkdist'); 

% plot the data distribution with the prototypes of the untrained network
figure;plot(X(1,:),X(2,:),'.g','markersize',10);
axis([-2 2 -2 2]);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off

% finally we train the network and see how their position changes
net.trainParam.epochs = 100;
net = train(net,X);
figure;plot3(X(1,:),X(2,:),X(3,:),'.g','markersize',10);
axis([-1 1 -1 1]);
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off

 