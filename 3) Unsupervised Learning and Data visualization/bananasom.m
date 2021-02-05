%% perform unsupervised learning with SOM  

% Marco Signoretto, March 2011

close all;
clear all;
clc;
% first we generate data uniformely distributed within two
% concentric cylinders

load("banana.mat")

figure;
hold on 
plot(X((Y==1),1),X((Y==1),2),'.g','markersize',10);
plot(X((Y==2),1),X((Y==2),2),'.r','markersize',10);

% we then initialize the SOM with hextop as topology function
% ,linkdist as distance function and gridsize 5x5x5
net = newsom(X',[5 5 5],'hextop','linkdist'); 

% plot the data distribution with the prototypes of the untrained network
figure;
hold on
plot(X((Y==1),1),X((Y==1),2),'.g','markersize',10);
plot(X((Y==2),1),X((Y==2),2),'.m','markersize',10);
plotsom(net.iw{1},net.layers{1}.distances)
xlim([-0.3 1.2])
ylim([0 1.4])
hold off

% finally we train the network and see how their position changes
net.trainParam.epochs = 100;
net = train(net,X');
figure;
hold on
plot(X((Y==1),1),X((Y==1),2),'.g','markersize',10);
plot(X((Y==2),1),X((Y==2),2),'.m','markersize',10);
axis([-1 1 -1 1]);
plotsom(net.iw{1},net.layers{1}.distances)
xlim([-0.3 1.2])
ylim([0 1.4])
hold off

 