import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import datasets
from sklearn import manifold

data = datasets.fetch_openml(
    'mnist_784',
    version = 1,
    return_X_y = True)

pixel_val, targets = data
targets = targets.astype(int)

#show image test
#single_img = pixel_val.iloc[1, :].values.reshape(28, 28)
#plt.imshow(single_img, cmap = 'gray')
#plt.show()

#t-Distributed Neighboring Embedding
tsne = manifold.TSNE(n_components = 2, random_state = 42)
transformed_data = tsne.fit_transform(pixel_val.iloc[:3000, :])

tsne_df = pd.DataFrame(
        np.column_stack((transformed_data, targets.iloc[:3000])),
        columns = ["x", "y", "targets"])
tsne_df["targets"] = tsne_df.targets.astype(int)

grid = sns.FacetGrid(tsne_df, hue = "targets", height = 8)
grid.map(plt.scatter, 'x', 'y').add_legend()
plt.show()
