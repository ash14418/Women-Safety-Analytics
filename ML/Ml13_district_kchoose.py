import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
os.makedirs("figures",exist_ok=True)
df=pd.read_csv("data/processed/district_scaled.csv")
X=df.drop(columns=["state_name","district_name"])
ks=range(2,11)
inertia=[]
sil=[]
for k in ks:
    km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X,km.labels_))
    print(k,round(km.inertia_,1),round(sil[-1],3))
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(list(ks),inertia,marker="o")
ax[0].set_title("Elbow method")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks),sil,marker="o",color="green")
ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k")
ax[1].set_ylabel("Score")
plt.tight_layout()
plt.savefig("figures/choose_k.png",dpi=300)
plt.close()
print("best k by silhouette:",list(ks)[sil.index(max(sil))])