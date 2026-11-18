import pandas as pd
import os
df=pd.read_csv("data/processed/district_2022_selected.csv")
groups={
    "rape_attempts":["murder_with_rape_gang_rape","rape_women_above_18","rape_girls_below_18","attempt_to_commit_rape_above_18","attempt_to_commit_rape_girls_below_18"],
    "assault_modesty":["assault_on_womenabove_18","assault_on_women_below_18","insult_to_the_modesty_of_women_above_18","insult_to_the_modesty_of_women_below_18","indecent_representation_of_women"],
    "kidnapping_trafficking":["kidnapping_and_abduction","kidnapping_abduction_in_order_to_murder","kidnp_and_abductn_of_women_above_18_for_marrg","kidnp_and_abductn_of_girls_below_18_for_marrg","kidnapping_and_abduction_of_women_others","procuration_of_minor_girls","importation_of_girls_from_foreign","human_trafficking","selling_of_minor_girls","buying_of_minor_girls","procuring_inducing_children_for_the_sake_of_prostitution","detaining_a_prsn_in_premises_where_prost_is_carried","prostitution_in_or_in_the_vicinity_of_public_places","seducing_or_soliciting_for_purpose_of_prostitution","other_sections_under_itp_act"],
    "domestic_dowry":["dowry_deaths","cruelty_by_husband_or_his_relatives","dowry_prohibition","protection_of_women_from_domestic_violence","abetment_to_suicide_of_women"],
    "cyber":["publshng_or_transmitting_of_sexually_explicit_mtrl","other_women_centric_cyber_crimes"],
    "pocso":["child_rape","sexual_assault_of_children","child_sexual_harassment","use_of_child_for_pornography","offences_of_pocso_act","pocso_act_unnatural_offences"],
    "other_violence":["miscarriage","acid_attack","attempt_to_acid_attack"]
}
#every crime rate column must belong to exactly one group, or the group rates would not add up to the total
assigned=[c+"_rate" for cols in groups.values() for c in cols]
all_rates=[c for c in df.columns if c.endswith("_rate") and c!="total_cases_rate"]
print("not assigned:",set(all_rates)-set(assigned))
print("assigned twice:",len(assigned)-len(set(assigned)))
for g,cols in groups.items():
    df["grp_"+g]=df[[c+"_rate" for c in cols]].sum(axis=1)
grp_cols=["grp_"+g for g in groups]
#the old check (group rates add up to the total rate) could never fail, because total_cases is itself the sum of all crime columns
#real check: 2022 district totals summed by state should match the state totals NCRB published (from Ml02)
state_file="data/processed/state_overview_2022_24.csv"
if os.path.exists(state_file):
    raw=pd.read_csv("data/processed/crime_against_women_clean.csv")
    raw=raw[raw["year"]==2022].copy()
    id_cols=["year","state_name","state_code","district_name"]
    raw["total_cases"]=raw[[c for c in raw.columns if c not in id_cols]].sum(axis=1)
    cmp=pd.concat([raw.groupby("state_name")["total_cases"].sum(),pd.read_csv(state_file).set_index("state_ut")["cases_2022"]],axis=1,keys=["district_sum","ncrb_state"]).dropna()
    cmp["ratio"]=cmp["district_sum"]/cmp["ncrb_state"]
    overall=cmp["district_sum"].sum()/cmp["ncrb_state"].sum()
    print("district totals vs NCRB state totals, 2022:",round(overall*100,1),"%")
    print("states more than 2% off:")
    print(cmp[(cmp["ratio"]-1).abs()>0.02].round(3))
    assert abs(overall-1)<0.01,"district totals no longer match NCRB state totals"
else:
    print("skipped NCRB total check:",state_file,"not found (run Ml02 first)")
print(df[grp_cols].describe().round(1))
out=df[["year","state_name","district_name","pop_female_2011","total_cases","total_cases_rate"]+grp_cols]
out.to_csv("data/processed/district_groups.csv",index=False)
print("saved",out.shape)