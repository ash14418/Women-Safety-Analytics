import pandas as pd
from sklearn.ensemble import IsolationForest
scaled=pd.read_csv("data/processed/district_scaled.csv")
clusters=pd.read_csv("data/processed/district_clusters.csv")
features=[c for c in scaled.columns if c.startswith("grp_")]
X=scaled[features]
#contamination=0.05 tells the model to flag the 5% most unusual districts
iso=IsolationForest(n_estimators=200,contamination=0.05,random_state=42)
iso.fit(X)
out=clusters[["state_name","district_name","total_cases_rate","cluster"]].copy()
out["anomaly"]=iso.predict(X)==-1
#lower score means more unusual
out["anomaly_score"]=iso.decision_function(X)
#the crime group furthest from average explains why a district was flagged
driver=X.abs().idxmax(axis=1)
out["main_driver"]=driver.str.replace("grp_","")
#positive = unusually high for that crime group, negative = unusually low (often zero reported cases)
out["driver_z"]=[X.loc[i,driver[i]] for i in X.index]
print("flagged:",out["anomaly"].sum(),"of",len(out))
print(out[out["anomaly"]].groupby("cluster").size())
top=out[out["anomaly"]].sort_values("anomaly_score").head(15)
print(top[["state_name","district_name","total_cases_rate","cluster","main_driver","driver_z"]].round(1).to_string(index=False))
out.sort_values("anomaly_score").to_csv("data/processed/district_anomalies.csv",index=False)
print("saved",out.shape)