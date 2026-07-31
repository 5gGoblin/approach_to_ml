import numpy as np
from sklearn import preprocessing

example = np.random.randint(1000, size = 1000000)

ohe = preprocessing.OneHotEncoder(sparse_output = False)
ohe_dense = ohe.fit_transform(example.reshape(-1, 1))
print(f"Size of dense array: {ohe_dense.nbytes}")

ohe = preprocessing.OneHotEncoder(sparse_output=True)
ohe_sparse = ohe.fit_transform(example.reshape(-1, 1))
print(f"Size of sparse array: {ohe_sparse.data.nbytes}")

total_size = (ohe_sparse.data.nbytes +
              ohe_sparse.indptr.nbytes +
              ohe_sparse.indices.nbytes)

print(f"Total sparse size: {total_size}")