import numpy as np
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt

# Generating a synthetic signal
np.random.seed(42)
n_samples = 200
n_features = 100
n_components = 30  # Number of atoms in the dictionary

time = np.linspace(0, 8, n_samples)
signals = np.zeros((n_samples, n_features))
for i in range(n_samples):
    signals[i] = np.sin(3 * time[i]) * np.random.randn(n_features)

# Adding noise to the signals
noise = 0.5 * np.random.randn(n_samples, n_features)
signals += noise

# Creating the dictionary using K-SVD
dico = DictionaryLearning(n_components=n_components, transform_algorithm='omp', random_state=42)
dictionary = dico.fit(signals).components_

# Obtaining sparse representation
code = np.dot(signals[0], dictionary.T)

# Plotting original signal and its sparse representation
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title('Original Signal')
plt.plot(signals[0], label='Original')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Sparse Representation')
plt.stem(code)
plt.xlabel('Basis Index')
plt.ylabel('Coefficient Value')

plt.tight_layout()
plt.show()
