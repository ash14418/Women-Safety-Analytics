import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
os.makedirs("figures",exist_ok=True)
scaled=pd.read_csv("data/processed/district_scaled.csv")
clusters=pd.read_csv("data/processed/district_clusters.csv")
anom=pd.read_csv("data/processed/district_anomalies.csv")
profile=pd.read_csv("data/processed/cluster_profile.csv",index_col="cluster")
features=[c for c in scaled.columns if c.startswith("grp_")]
#PCA squeezes the 6 features into 2 axes so the clusters can be drawn on a flat chart
pca=PCA(n_components=2,random_state=42)
pts=pca.fit_transform(scaled[features])
print("variance explained:",pca.explained_variance_ratio_.round(3),"total:",round(pca.explained_variance_ratio_.sum(),3))
df=clusters[["state_name","district_name","cluster"]].copy()
df["pc1"]=pts[:,0]
df["pc2"]=pts[:,1]
df=df.merge(anom[["state_name","district_name","anomaly"]],on=["state_name","district_name"])
names={0:"0 Lowest reported crime",1:"1 Rape-led, low assault/POCSO",2:"2 Moderate-high, domestic-led",3:"3 Highest across groups"}
colors={0:"#4C9F70",1:"#E8A33D",2:"#4A7FB5",3:"#C0392B"}
plt.figure(figsize=(9,6))
for c in range(4):
    d=df[df["cluster"]==c]
    plt.scatter(d["pc1"],d["pc2"],s=18,alpha=0.7,color=colors[c],label=names[c])
a=df[df["anomaly"]]
plt.scatter(a["pc1"],a["pc2"],s=90,facecolors="none",edgecolors="black",linewidths=1.2,label="Anomaly")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("District clusters (PCA view) with anomalies circled")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/clusters_pca.png",dpi=300)
plt.close()
cols=[c for c in profile.columns if c not in ["districts","total_rate"]]
ax=profile[cols].plot(kind="bar",figsize=(10,5))
ax.set_xticklabels([f"Cluster {i}\n({int(profile.loc[i,'districts'])} districts)" for i in profile.index],rotation=0)
plt.title("Average reported cases per lakh women, by cluster and crime group")
plt.ylabel("Cases per lakh women")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/cluster_profile.png",dpi=300)
plt.close()
print("saved 2 figures")