data = (randn(50,500));
[coeff, score, latent, tsquared, explained, mu] = pca(data');

%[eigvals,eigvec] = linearpca(data');

[~,n_components] = max(cumsum(explained) >= 95);

mse_mat = zeros(1,size(data,1));

for i=1:size(data,1)
    X_reconstruct = bsxfun(@plus, score(:,1:i) * coeff(:,1:i).', mu);
    X_reconstruct = X_reconstruct';
    mse_mat(:,i)=sqrt(mean(mean((data-X_reconstruct).^2)))
end

%Cumexplained mat 
cumexplained = cumsum(explained);
cumunexplained = 100 - cumexplained;

% Plot MSE
figure
plot(1:50, mse_mat, '-o','MarkerFaceColor','y','MarkerEdgeColor','k');
grid on;
xlabel('Number of Components');
ylabel('MSE')
title('MSE reduction (Gaussian)')

% Plot eigenvalues
figure('Name','eigenvalues of the covariance matrix ','NumberTitle','off');
plot([1:length(latent)],latent,'-o','MarkerFaceColor','r','MarkerEdgeColor','k')
ylabel('Eigenvalues')
xlabel('Subscript of Eigenvalues')