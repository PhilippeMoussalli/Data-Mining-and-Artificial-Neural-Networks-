%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML110
% Project Title: Implementation of DBSCAN Clustering in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clc;
clear;
close all;

%% Load Data

data=load('rings');
X=data.X;


%% Run DBSCAN Clustering Algorithm
ep=0;
for i=1:5
    ep=ep+0.5;
    pnts = 5;
        for j=1:4
        epsilon=ep;
        MinPts=pnts;
        IDX=DBSCAN(X,epsilon,MinPts);
       

        %% Plot Results
        figure
        PlotClusterinResult(X, IDX);
        title(['DBSCAN Clustering (\epsilon = ' num2str(epsilon) ', MinPts = ' num2str(MinPts) ')']);
        pnts=pnts+5;
        end
end

