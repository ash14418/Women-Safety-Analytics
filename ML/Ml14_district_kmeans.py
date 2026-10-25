import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
scaled=pd.read_csv("data/processed/district_scaled.csv")
groups=pd.read_csv("data/processed/district_groups.csv")
X=scaled.drop(columns=["state_name","district_name"])
k=4
km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X)
groups["cluster"]=km.labels_
#cluster numbers are arbitrary, so they are renumbered from lowest to highest average total rate to keep labels stable
order=groups.groupby("cluster")["total_cases_rate"].mean().sort_values().index
groups["cluster"]=groups["cluster"].map({old:new for new,old in enumerate(order)})
print("silhouette:",round(silhouette_score(X,groups["cluster"]),3))
cols=[c for c in groups.columns if c.startswith("grp_") and c!="grp_other_violence"]
profile=groups.groupby("cluster").agg(districts=("cluster","size"),total_rate=("total_cases_rate","mean"),**{c.replace("grp_",""):(c,"mean") for c in cols}).round(1)
print(profile.to_string())
#the 5 districts with the highest total rate in each cluster, to help name the clusters
for c in range(k):
    top=groups[groups["cluster"]==c].nlargest(5,"total_cases_rate")
    print("\ncluster",c,":",", ".join(top["district_name"]+" ("+top["state_name"]+")"))
profile.to_csv("data/processed/cluster_profile.csv")
groups.to_csv("data/processed/district_clusters.csv",index=False)
print("saved",groups.shape)