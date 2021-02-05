# Data Mining and Artificial Neural Networks
Exercises sessions and [report](ANN_report.pdf) for the course “Data Mining and Artificial Neural Networks” taught at KU Leuven. The course focused on gaining insight and hands-on expertise. The tackled subjects are split into the following sections: 

## 1) Training Algorithms and Generalization  
  This section introduces the use of neural networks as general models for function estimation. From their simplest form, the perceptron, to more complex structures. The goal was to gain a sufficient understanding of the mathematics of the model, its power and its limitations, its estimation, and its training as well as the precise influence of the model parameters. This section includes:  
    
   * Approximating a non-linear function with a Neural Network and a one layered perceptron to gain insight about the representational capabilities of each architecture  
   * Training a Multi-layered network using backpropagation with different optimization functions and comparing their performance. Those include: Quasi-Newtonian, Conjugate gradient, Gradient descent, Levenberg-Marquard  
   * Exploring different activation functions  
   * Bayesian inference for regularization of network weights  
   * The curse of dimensionality 
   
## 2) Time Series Prediction and Data Classification  
  This section focuses on applying the feedforward neural networks to typical machine learning tasks,
such as time-series prediction and classification. This section includes:
  
   * Time series forecasting for the Santa Fe laser dataset and the Shanghai dataset (air pollution prediction). 
   * Binary classification of Breast Cancer from the Wisconsin (Diagnostic) dataset with a multi-layered perceptron trained and validated on different networks and optimization functions to derive the optimal hyperparameters.  
   * Using Automatic Relevance Determination on the Wisconsin dataset to determine the input attributes that are the most relevant to the prediction task (dimensionality reduction)
   
   ## 3) Unsupervised Learning and Data visualization 
  This section investigates unsupervised learning tasks, including PCA analysis, k -means clustering,
  density-based methods such as GMM, kernel density estimation, and vector quantization and self-organizing maps. It includes: 

   * PCA as a dimensionality reduction method for correlated data 
   * Compression and reconstruction of handwritten digits (MNIST dataset) using PCA 
   * Unsupervised classification through K-means clustering and density-based methods: Gaussian mixture models (GMM) and Kernel density estimation (KDE)  
   * Self-Organizing maps for dimensionality reduction and data representation
   
   ## 4) Autoencoders and Convolutional Neural Networks  
  This section was aimed at gaining hands-on practice on slightly more complicated neural networks namely, Autoencoders and Convolutional Neural networks.      Different variants of these networks were explored and there with a focus on their ability to extract powerful features to perform various machine        learning tasks. In particular, the aim is to understand how such architectures are trained and the corresponding arithmetic involved. It includes:

   * Stacked autoencoder for feature extraction and non-linear dimensionality reduction
   * Variational autoencoder for generative models 
   * Training Convolutional Neural Network for classification of handwritten digits 
