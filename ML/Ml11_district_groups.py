import pandas as pd
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
#check: the group rates should add up to the total rate
print("max gap vs total rate:",(df[grp_cols].sum(axis=1)-df["total_cases_rate"]).abs().max())
print(df[grp_cols].describe().round(1))
out=df[["year","state_name","district_name","pop_female_2011","total_cases","total_cases_rate"]+grp_cols]
out.to_csv("data/processed/district_groups.csv",index=False)
print("saved",out.shape)