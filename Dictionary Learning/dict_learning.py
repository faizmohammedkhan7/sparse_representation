## load library
import numpy as np
from sklearn.decomposition import DictionaryLearning

# create input data
X = np.array([[2, -1, 0, -1], [-1, 0, 0, 1], [-1, 0, -1, 2]])
print(f'X matrix shape : {X.shape}')
X
# init dictionary learning model 
# NOTE : we already known the rank of X = 3; used 'orthogonal matching pursuit algorithm'
dict_learner = DictionaryLearning(n_components=3, transform_algorithm='omp', random_state=42)

# fit the input data to the dictionary learning model
dict_learner.fit(X)

dict_learner.get_params()

R = dict_learner.transform(X)

np.sum(R == 0)

R = dict_learner.transform(X)
print(f'R matrix shape : {R.shape}')
print(f'# of zero entries in the R : {np.sum(R == 0)} \npercentage of zero entries : {np.mean(R == 0)}')
R

# getting D: dictionary matrix
R_inv = np.linalg.inv(R)
D = np.dot(R_inv, X)
print(f'D matrix shape : {D.shape}')
D

# proof of X = DR
np.dot(R, D)