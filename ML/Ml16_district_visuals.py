import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
scaled = pd.read_csv("data/processed/district_scaled.csv")
clusters = pd.read_csv("data/processed/district_clusters.csv")
anom = pd.read_csv("data/processed/district_anomalies.csv")
profile = pd.read_csv("data/processed/cluster_profile.csv", index_col="cluster")
features = [c for c in scaled.columns if c.startswith("grp_")]

# the cluster names below were written by hand for k=4; stop with a clear error if a re-run changes what the clusters mean
gcols = [c for c in profile.columns if c not in ["districts", "total_rate"]]
assert list(profile.index) == [0, 1, 2, 3], "expected 4 clusters, so the hardcoded names no longer apply"
assert profile["total_rate"].is_monotonic_increasing, "clusters are not ordered by total rate"
assert (profile.loc[3, gcols] == profile[gcols].max()).sum() >= 5, "cluster 3 is no longer highest across groups"
assert profile.loc[1, "rape_attempts"] > profile.loc[2, "rape_attempts"], "cluster 1 is no longer rape-led"
assert profile.loc[1, "assault_modesty"] < profile.loc[2, "assault_modesty"], "cluster 1 no longer has low assault"
assert profile.loc[1, "pocso"] == profile["pocso"].min(), "cluster 1 no longer has lowest POCSO"
assert profile.loc[2, gcols].idxmax() == "domestic_dowry", "cluster 2 is no longer domestic-led"

# PCA via numpy SVD (replaces sklearn, which is blocked by Application Control on this machine)
X = scaled[features].to_numpy(dtype=float)
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
# fix component signs deterministically (largest-magnitude loading positive), like sklearn's svd_flip convention
top = np.abs(Vt[:2]).argmax(axis=1)
signs = np.sign(Vt[:2][np.arange(2), top])
signs[signs == 0] = 1
pts = (Xc @ Vt[:2].T) * signs
var_ratio = (S ** 2) / (S ** 2).sum()
print("variance explained:", var_ratio[:2].round(3), "total:", round(var_ratio[:2].sum(), 3))

df = clusters[["state_name", "district_name", "cluster"]].copy()
df["pc1"] = pts[:, 0]
df["pc2"] = pts[:, 1]
df = df.merge(anom[["state_name", "district_name", "anomaly"]], on=["state_name", "district_name"])

names = {0: "0 Lowest reported crime", 1: "1 Rape-led, low assault/POCSO", 2: "2 Moderate-high, domestic-led", 3: "3 Highest across groups"}
colors = {0: "#4C9F70", 1: "#E8A33D", 2: "#4A7FB5", 3: "#C0392B"}

plt.figure(figsize=(9, 6))
for c in profile.index:
    d = df[df["cluster"] == c]
    plt.scatter(d["pc1"], d["pc2"], s=18, alpha=0.7, color=colors[c], label=names[c])
a = df[df["anomaly"]]
plt.scatter(a["pc1"], a["pc2"], s=90, facecolors="none", edgecolors="black", linewidths=1.2, label="Anomaly")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("District clusters (PCA view) with anomalies circled")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/clusters_pca.png", dpi=300)
plt.close()

cols = [c for c in profile.columns if c not in ["districts", "total_rate"]]
ax = profile[cols].plot(kind="bar", figsize=(10, 5))
ax.set_xticklabels([f"Cluster {i}\n({int(profile.loc[i, 'districts'])} districts)" for i in profile.index], rotation=0)
plt.title("Average reported cases per lakh women, by cluster and crime group")
plt.ylabel("Cases per lakh women")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/cluster_profile.png", dpi=300)
plt.close()
print("saved 2 figures")