%This example explores the effects of specifying different options 
%for covariance structure and initial conditions when you perform 
%GMM clustering.

%Load Fisher's iris data set. Consider clustering the sepal measurements, 
%and visualize the data in 2-D using the sepal measurements.

[n,p] = size(X);

plot(X(:,1),X(:,2),'.','MarkerSize',15);
title('Fisher''s Iris Data Set');
xlabel('Sepal length (cm)');
ylabel('Sepal width (cm)');

%The number of components k in a GMM determines the number of subpopulations, 
%or clusters. In this figure, it is difficult to determine if two, three, 
%or perhaps more Gaussian components are appropriate. 
%A GMM increases in complexity as k increases.

%% Specify Different Covariance Structure Options

%Each Gaussian component has a covariance matrix. Geometrically, the covariance 
%structure determines the shape of a confidence ellipsoid drawn over a cluster. 
%You can specify whether the covariance matrices for all components are diagonal 
%or full, and whether all components have the same covariance matrix. 
%Each combination of specifications determines the shape and orientation 
%of the ellipsoids.

%Specify three GMM components and 1000 maximum iterations for the EM algorithm. 
%For reproducibility, set the random seed.

%rng(3);
k = 3; % Number of GMM components
options = statset('MaxIter',1000);

%Specify covariance structure options.
Sigma = {'diagonal','full'}; % Options for covariance matrix type
nSigma = numel(Sigma);

SharedCovariance = {false,true}; % Indicator for identical or nonidentical covariance matrices
SCtext = {'false','true'};
nSC = numel(SharedCovariance);

%Create a 2-D grid covering the plane composed of extremes of the measurements. 
%You will use this grid later to draw confidence ellipsoids over the clusters.
d = 500; % Grid length
x1 = linspace(min(X(:,1))-2, max(X(:,1))+2, d);
x2 = linspace(min(X(:,2))-2, max(X(:,2))+2, d);
[x1grid,x2grid] = meshgrid(x1,x2);
X0 = [x1grid(:) x2grid(:)];

%Specify the following:
%-For all combinations of the covariance structure options, fit a GMM with three components.
%-Use the fitted GMM to cluster the 2-D grid.
%-Obtain the score that specifies a 99% probability threshold for each confidence region. 
%This specification determines the length of the major and minor axes of the ellipsoids.
%-Color each ellipsoid using a similar color as its cluster.
threshold = sqrt(chi2inv(0.99,2));
count = 1;
for i = 1:nSigma
    for j = 1:nSC
        gmfit = fitgmdist(X,k,'CovarianceType',Sigma{i}, ...
            'SharedCovariance',SharedCovariance{j},'Options',options); % Fitted GMM
        clusterX = cluster(gmfit,X); % Cluster index 
        mahalDist = mahal(gmfit,X0); % Distance from each grid point to each GMM component
        % Draw ellipsoids over each GMM component and show clustering result.
        subplot(2,2,count);
        h1 = gscatter(X(:,1),X(:,2),clusterX);
        hold on
            for m = 1:k
                idx = mahalDist(:,m)<=threshold;
                Color = h1(m).Color*0.75 - 0.5*(h1(m).Color - 1);
                h2 = plot(X0(idx,1),X0(idx,2),'.','Color',Color,'MarkerSize',1);
                uistack(h2,'bottom');
            end    
        plot(gmfit.mu(:,1),gmfit.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
        title(sprintf('Sigma is %s\nSharedCovariance = %s',Sigma{i},SCtext{j}),'FontSize',8)
        legend(h1,{'1','2','3'})
        hold off
        count = count + 1;
    end
end