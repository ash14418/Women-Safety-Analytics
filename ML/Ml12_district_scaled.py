import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
df=pd.read_csv("data/processed/district_groups.csv")
#grp_other_violence is almost always 0, so it adds noise and is left out
features=["grp_rape_attempts","grp_assault_modesty","grp_kidnapping_trafficking","grp_domestic_dowry","grp_cyber","grp_pocso"]
X=df[features]
print("skew before:")
print(X.skew().round(2))
#rates are skewed (a few districts are very high); log1p compresses the long tail and keeps zeros valid
X_log=np.log1p(X)
print("skew after log1p:")
print(X_log.skew().round(2))
#KMeans uses distances, so every feature needs the same scale (mean 0, std 1)
scaler=StandardScaler()
X_scaled=pd.DataFrame(scaler.fit_transform(X_log),columns=features,index=df.index)
print(X_scaled.describe().loc[["mean","std"]].round(2))
out=pd.concat([df[["state_name","district_name"]],X_scaled],axis=1)
out.to_csv("data/processed/district_scaled.csv",index=False)
print("saved",out.shape)