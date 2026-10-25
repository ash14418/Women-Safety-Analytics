import pandas as pd
import matplotlib.pyplot as plt
import os
os.makedirs("figures",exist_ok=True)
df=pd.read_csv("data/processed/model_results.csv")
#the baseline row is the reference every model is compared against
base=df.loc[df["Model"]=="Naive (previous year)","MAE"].iloc[0]
ax=df.plot(x="Model",y="MAE",kind="bar",legend=False,figsize=(9,5))
ax.axhline(base,color="red",linestyle="--",label="Baseline MAE")
for c in ax.containers:ax.bar_label(c,fmt="%.0f")
plt.title("Model MAE vs baseline")
plt.ylabel("MAE (cases)")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.savefig("figures/baseline_mae.png",dpi=300)
plt.close()
#each dot is a district; the red line is where the baseline prediction (this year = last year) would be exactly right
ml=pd.read_csv("data/processed/ml_dataset.csv")
test=ml[ml["year"]>2020]
plt.figure(figsize=(6,6))
plt.scatter(test["previous_year_cases"],test["total_cases"],s=10,alpha=0.5)
top=max(test["previous_year_cases"].max(),test["total_cases"].max())
plt.plot([0,top],[0,top],color="red",linestyle="--",label="Baseline (this year = last year)")
plt.xlabel("Last year cases")
plt.ylabel("Actual cases")
plt.title("Baseline vs actual, 2021-2022")
plt.legend()
plt.tight_layout()
plt.savefig("figures/baseline_scatter.png",dpi=300)
plt.close()
print("saved")