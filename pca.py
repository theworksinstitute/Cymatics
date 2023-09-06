import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#df = pd.read_excel("Baseline_ALL_202303.xlsx", sheet_name="ALL")
df = pd.read_excel("combined_data.xlsx", sheet_name="Munka1")
#df = pd.read_excel("cymatics_ez_water_experiment_ALL.xlsx", sheet_name="Sheet1")

mycolumns = ["Duration [min]", "Humidity [%]", "Air pressure [mb]", "Water temp. [⁰C]", "Air temp. [⁰C]", "Moon illumination"]
X = df[mycolumns].to_numpy()
y = df["Symm1"].to_numpy()

# Standardize the predictor variables to have mean=0 and variance=1
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Perform PCA on the standardized data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# Plot the first two principal components against the target variable
import matplotlib.pyplot as plt

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.savefig("pca.png")
