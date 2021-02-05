load( "threes.mat",'-ascii') 

%Plotting of the image 
figure
colormap('gray')
imagesc(reshape(threes(5,:),16,16),[0,1])

%PCA
data = threes;
[coeff, score, latent, tsquared, explained, mu] = pca(data);

%Project only on the first 4 components 
coeff_pca = coeff(:,[1:4]);
X3_pca =coeff_pca'*data';
[~,n_components] = max(cumsum(explained) >= 95);
%Reconstructions
% X_reconstruct = bsxfun(@plus, X3_pca' * coeff_pca.', mu);

%It seems like  adding the mean back again to the reconstructed dataset is making MSE
% higher when the largest eigenvalues are used --> Obstain from using it in
% this example

X_reconstruct= X3_pca' * coeff_pca.';

sqrt(mean(mean((data-X_reconstruct).^2)))


%Plotting
figure
colormap('gray')
imagesc(reshape(X_reconstruct(5,:),16,16),[0,1])

% Plot eigenvalues
figure('Name','eigenvalues of the covariance matrix ','NumberTitle','off');
plot([1:length(latent)],latent,'-o','MarkerFaceColor','r','MarkerEdgeColor','k')
ylabel('Eigenvalues')
xlabel('Subscript of Eigenvalues')


% Try for different reconstructions (Up to 50)
k=50;
mse_mat = zeros(1,k);

for i=1:k
    coeff2 = coeff(:,[1:i]);
    X3_pca =coeff2'*data';
    X_reconstruct= X3_pca' * coeff2.';
    mse_mat(:,i)=sqrt(mean(mean((data-X_reconstruct).^2)));
end


cumexplained = cumsum(latent);

figure
hold on 
plot([1:50],mse_mat,'-o','MarkerFaceColor','r','MarkerEdgeColor','k')
ylabel('MSE')
xlabel('Subscript of Eigenvalues')

figure
hold on 
plot([1:50],cumexplained([1:50],:),'-o','MarkerFaceColor','y','MarkerEdgeColor','k')
ylabel('Cumulative sum of eigenvalues')
xlabel('Subscript of Eigenvalues')