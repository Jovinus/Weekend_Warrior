# %% Import package to use
import datatable
import pandas as pd
import numpy as np
import os
from glob import glob
from IPython.display import display
pd.set_option('display.max_columns', None)

# %% Processing the dataset
DATAPATH = "../../VO2max_Prediction/Data/processed_whole_set.csv"
df_init = datatable.fread(DATAPATH, na_strings=['','NA'], encoding='utf-8-sig').to_pandas()
df_init['SM_DATE'] = df_init['SM_DATE'].astype('datetime64')

## Male Only
# df_init = df_init[df_init['sex'] == 0].reset_index(drop=True)
display(df_init.head(1))

# %% 
print("Inclusion n = ", len(df_init['HPCID'].unique()))

#### Exclude outlier
for i in df_init[['SM3631', 'SM0104', 'SM3720']].columns:
    if i == 'SM3720':
        df_init = df_init[(df_init[i] <= df_init[i].quantile(0.999)) & (df_init[i] >= df_init[i].quantile(0.0001))]
    else:
        df_init = df_init[(df_init[i] >= df_init[i].quantile(0.005)) & (df_init[i] <= df_init[i].quantile(0.995))]

print("Number of population in set = {}".format(len(set(df_init['HPCID']))))

#### Exclude patient have and had CVD
df_init['CVD'] = np.where(df_init[['Stroke', 'Angina', 'MI', 'Cancer']].isin([1]).any(axis=1), 1, 0)

# %%
df_selected = df_init.copy()
print("Number of healthy population = {}".format(len(set(df_selected['HPCID']))))

# %% Change columnn Name

columns_to_use = ['SM_DATE', 'CDW_NO', 'HPCID', 'sex', 'AGE', 'SM0104', 'SM0101', 
                'SM0102', 'SM316001', 'MVPA', 'SM3631', 'Smoke', 'SM3720', 'SM0106', 'SM0111', 
                'SM0112', 'SM0126', 'SM0151', 'SM0152', 'SM0153', 'SM0154', 'SM0155', 'SM3140', 
                'SM3150', 'SM3170', 'CRP', 'CHOLESTEROL', 'TG', 'max_heart_rate', 'BMI_cal', 'BL3118', 
                'ASMI', 'VO2max', 'death', 'delta_time', 'Diabetes', 'Hypertension', 'HTN_med', 
                'Hyperlipidemia', 'Hepatatis', 'ALC', 'BL3142', 'BL314201', 'MBP', 'SM0600SBP', 
                'SM0600DBP', 'MED_HYPERTENSION', 'MED_HYPERLIPIDEMIA', "RER_over_gs", 'METs_week', 
                'OVERALL_PHYSICAL_ACTIVITY', 'PHY_FREQ', 'PHY_DURATION', 'CVD']

columns_to_rename = {'SM0104':'percentage_fat', 'SM0101':'Height', 
                    'SM0102':'Weight', 'SM316001': 'BMI', 
                    'SM3631':'rest_HR', 'SM3720':'CRF', 
                    'SM0106':'비만도', 'SM0111':'Muscle_mass', 
                    'SM0112':'복부지방율', 'SM0126':'부종검사', 
                    'SM0151':'Muscle_mass(RA)', 'SM0152':'Muscle_mass(LA)', 
                    'SM0153':'Muscle_mass(BODY)', 'SM0154':'Muscle_mass(RL)', 
                    'SM0155':'Muscle_mass(LL)', 'SM3140':'체지방량', 
                    'SM3150':'체수분량', 'SM3170':'제지방량', 'BL3142':"HDL_C", 'BL314201':'LDL_C',
                    "SM0600SBP":"SBP", "SM0600DBP":"DBP", "BL3118":'Glucose, Fasting', 
                    'OVERALL_PHYSICAL_ACTIVITY':'PA', 'PHY_FREQ':'PA_FREQ', 'PHY_DURATION':'PA_DUR'}


df_selected_eq = df_selected[columns_to_use].rename(columns = columns_to_rename)
df_selected.rename(columns= columns_to_rename, inplace=True)


# %%

df_cac = pd.read_csv('../../VO2max_Prediction/Data/request/vo2max_cac.csv')
df_carotid = pd.read_csv('../../VO2max_Prediction/Data/request/vo2max_carotid.csv')
df_pwv = pd.read_csv('../../VO2max_Prediction/Data/request/vo2max_pwv_abi.csv')

# %%

df_cac['SM_DATE'] = df_cac['SM_DATE#2'].astype('datetime64')
df_pwv['SM_DATE'] = df_pwv['처방일자#4'].astype('datetime64')
df_carotid['SM_DATE'] = df_carotid['건진일자#2'].astype('datetime64')

#%%

df_cac = df_cac[['환자번호#1', 'SM_DATE', 'AJ_130_SCORE#3', 'VOLUME_SCORE#4']]
df_cac.rename(columns={'VOLUME_SCORE#4': 'Volume_Score', 'AJ_130_SCORE#3': 'CAC'}, inplace=True)

# %%

df_carotid['mean_IMT'] = df_carotid.loc[:, 'Carotid US CCA : IMT(Rt)#5':'Carotid US CCA : IMT(Lt)#6'].mean(axis=1)
mask_carotid = (df_carotid.loc[:, 'Carotid US CCA : IMT(Rt)#5':'Carotid US CCA : IMT(Lt)#6'].notnull().sum(axis=1) >= 2)
df_carotid = df_carotid[mask_carotid][['환자번호#1', 'SM_DATE', 'mean_IMT']].reset_index(drop=True)
# %%

df_pwv = pd.pivot_table(values='검사결과수치값#7', columns='검사명#6', index=['환자번호#1', 'SM_DATE'], data=df_pwv, aggfunc=lambda x: x).reset_index(drop=False)
df_pwv.columns.name = None

# %%

def split_data(input_x, option=0):
    
    if option != 0:
        option = 1
        
    splitted = input_x.split('/')
    
    try:
        return float(splitted[option])
    
    except (ValueError, IndexError):
        return np.nan

# %%

df_pwv['baPWV_Rt'] = df_pwv['baPWV(Rt/Lt)'].apply(lambda x: split_data(input_x=x, option=0))
df_pwv['baPWV_Lt'] = df_pwv['baPWV(Rt/Lt)'].apply(lambda x: split_data(input_x=x, option=1))
df_pwv['ABI_Rt'] = df_pwv['ABI(Rt/Lt)'].apply(lambda x: split_data(input_x=x, option=0))
df_pwv['ABI_Lt'] = df_pwv['ABI(Rt/Lt)'].apply(lambda x: split_data(input_x=x, option=1))

df_pwv['baPWV'] = df_pwv[['baPWV_Lt', 'baPWV_Rt']].mean(axis=1)
df_pwv['ABI'] = df_pwv[['ABI_Lt', 'ABI_Rt']].mean(axis=1)

df_pwv = df_pwv[['환자번호#1', 'SM_DATE' ,'baPWV', 'ABI']]
# %%
df_results = pd.merge(df_pwv, df_carotid, on=['환자번호#1', 'SM_DATE'], how='outer')
df_results = pd.merge(df_results, df_cac, on=['환자번호#1', 'SM_DATE'], how='outer')
df_results.rename(columns={'환자번호#1':'ID'}, inplace=True)
# %%
display(df_results)

# %%
df_final = pd.merge(df_selected_eq, df_results, 
                    left_on=['CDW_NO', 'SM_DATE'], 
                    right_on=['ID', 'SM_DATE'], how='left')


# %%
df_final = df_final.assign(pa_freq_tmp = lambda x: x['PA_FREQ'].map({0:0, 1:1.5, 2:3.5, 3:6}),
                           pa_dur_tmp = lambda x: x['PA_DUR'].map({0:0, 1:20, 2:30, 3:50, 4:60}),
                           week_pa_time = lambda x: x['pa_freq_tmp'] * x['pa_dur_tmp'])

# %%
df_final['PAP'] = np.nan
df_final.loc[(df_final['PA'] == 2) & (df_final['week_pa_time'] >= 150) & (df_final['PA_FREQ'] == 1), 'PAP'] = "weekend_warrior"
df_final.loc[(df_final['PAP'].isnull()) & (df_final['PA'] == 2) & (df_final['week_pa_time'] >= 150) & (df_final['PA_FREQ'] > 1), 'PAP'] = "regularly_active"

df_final.loc[(df_final['PAP'].isnull()) & (df_final['PA'] == 3) & (df_final['week_pa_time'] >= 75) & (df_final['PA_FREQ'] == 1), 'PAP'] = "weekend_warrior"
df_final.loc[(df_final['PAP'].isnull()) & (df_final['PA'] == 3) & (df_final['week_pa_time'] >= 75) & (df_final['PA_FREQ'] > 1), 'PAP'] = "regularly_active"

df_final.loc[(df_final['PAP'].isnull()) & (df_final['PA'] >= 1) & (df_final['week_pa_time'] > 0), 'PAP'] = "insufficiently_active"

df_final.loc[df_final['PAP'].isnull(), 'PAP'] = "inactive"

# %%
df_final.to_csv("../data/fitness_dataset.csv", index=False, encoding='utf-8-sig')

print(f"Final Data N = {len(set(df_final['CDW_NO']))}")
# %%
