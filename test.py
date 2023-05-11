import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Read the csv file
df = pd.read_csv("iris.csv")

# Create a figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Perform PCA
pca = PCA(n_components=3)
pca.fit(df)

# Transform the data
X = pca.transform(df)

# Plot the data
ax.scatter(X[:, 0], X[:, 1], X[:, 2])

plt.show()
