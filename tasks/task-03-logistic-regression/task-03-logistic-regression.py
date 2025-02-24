import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LogisticNeuron:
    def __init__(self, input_dim, learning_rate=0.1, epochs=1000):
        """
        Initialize the neuron with random weights and a bias term.
        """
        ### START CODE HERE ###
        
        # Inicializar pesos aleatórios pequenos e bias como zero
        self.weights = np.random.randn(input_dim)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []  # Lista para armazenar a perda durante o treinamento
        
        ### END CODE HERE ###
    
    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.
        """
        ### START CODE HERE ###
        
        return 1 / (1 + np.exp(-z))
        
        ### END CODE HERE ###
    
    def predict_proba(self, X):
        """
        Compute the probability of class 1 using the logistic function.
        """
        ### START CODE HERE ###
        
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
        
        ### END CODE HERE ###
    
    def predict(self, X):
        """
        Return binary predictions (0 or 1) based on probability.
        """
        ### START CODE HERE ###
        
        return (self.predict_proba(X) >= 0.5).astype(int)
        
        ### END CODE HERE ###
    
    def train(self, X, y):
        """
        Train the model using gradient descent.
        """
        ### START CODE HERE ###
        
        m = X.shape[0]  # Número de exemplos
        
        for epoch in range(self.epochs):
            # Calcular predições
            y_pred = self.predict_proba(X)

            # Calcular erro
            error = y_pred - y

            # Calcular gradientes
            dW = np.dot(X.T, error) / m  # Gradiente dos pesos
            dB = np.sum(error) / m  # Gradiente do bias

            # Atualizar pesos e bias
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

            # Calcular e armazenar perda (log loss)
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
            self.loss_history.append(loss)

        ### END CODE HERE ###

def generate_dataset():
    X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2.0)
    return X, y

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Logistic Regression Output')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

def plot_loss(model):
    plt.plot(model.loss_history, 'k.')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss over Training Iterations')
    plt.show()

# Generate dataset
X, y = generate_dataset()

# Train the model
neuron = LogisticNeuron(input_dim=2, learning_rate=0.1, epochs=100)
neuron.train(X, y)

# Plot decision boundary
plot_decision_boundary(neuron, X, y)

# Plot loss over training iterations
plot_loss(neuron)
