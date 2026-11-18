import pandas as pd
import os
import re
url="data/raw/Women_Safety_NCRB_2023_Data.xlsx"
year="2023"
output_folder="data/processed"
print("saving to:",os.path.abspath(output_folder))
os.makedirs(output_folder,exist_ok=True)
#one sheet per table (state, city, crime head, rape tables) plus a README
xls=pd.ExcelFile(url)
print(xls.sheet_names)
for sheet in xls.sheet_names:
    #README holds notes, not data
    if sheet=="README":
        continue
    df=pd.read_excel(url,sheet_name=sheet)
    #lowercase column names with underscores and no stray spaces
    df.columns=df.columns.str.strip().str.lower().str.replace(" ","_")
    #remove empty rows, empty columns and exact duplicates
    df=df.dropna(how="all")
    df=df.dropna(axis=1,how="all")
    df=df.drop_duplicates()
    #TOTAL rows are summaries and would be counted as extra states in modelling
    first_col=df.columns[0]
    df=df[~df[first_col].astype(str).str.upper().str.startswith("TOTAL")]
    #sheets like State_Overview_2021_23 only contain "23", so the year is matched as a whole number: 2023 or 23
    #sheets with no year in the name (rape tables) get the year added to avoid overwriting the 2024 files
    name=sheet.lower()
    if not re.search(r"(?<!\d)(%s|%s)(?!\d)"%(year,year[-2:]),name):
        name=name+"_"+year
    df.to_csv(os.path.join(output_folder,name+".csv"),index=False)
    print(sheet,"→",df.shape)