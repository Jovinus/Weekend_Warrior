# %%
from operator import index
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
# %%
df_orig = pd.read_csv("../data/fitness_dataset.csv")
df_orig.head()

print(len(set(df_orig['CDW_NO'])))

print(df_orig.groupby(['CDW_NO'])['sex'].head(1).value_counts())
# %%
study_set = df_orig.query('baPWV.notnull() & mean_IMT.notnull() & CAC.notnull() & AGE >= 40', engine='python')
study_set = study_set.groupby(['CDW_NO']).head(1)
# %%
study_set = study_set.assign(age_cat = lambda x: np.where(x['AGE'] >= 60, 2, np.where(x['AGE'] >= 50, 1, 0)),
                             cac_cut_0 = lambda x: np.where(x['CAC'] > 0, 1, 0),
                             cac_cut_100 = lambda x: np.where(x['CAC'] > 100, 1, 0),
                            #  cimt_75 = lambda x: np.where(x['mean_IMT'] > 0.8, 1, 0),
                            #  bapwv_75 = lambda x: np.where(x['mean_IMT'] > 0.8, 1, 0)
                             )

study_set_m = study_set.query("sex == False").reset_index(drop=True)
study_set_f = study_set.query("sex == True").reset_index(drop=True)
# %%
male_imt_cut_0 = study_set_m.query("age_cat == 0")['mean_IMT'].quantile(0.75)
male_imt_cut_1 = study_set_m.query("age_cat == 1")['mean_IMT'].quantile(0.75)
male_imt_cut_2 = study_set_m.query("age_cat == 2")['mean_IMT'].quantile(0.75)

male_pwv_cut_0 = study_set_m.query("age_cat == 0")['baPWV'].quantile(0.75)
male_pwv_cut_1 = study_set_m.query("age_cat == 1")['baPWV'].quantile(0.75)
male_pwv_cut_2 = study_set_m.query("age_cat == 2")['baPWV'].quantile(0.75)

female_imt_cut_0 = study_set_f.query("age_cat == 0")['mean_IMT'].quantile(0.75)
female_imt_cut_1 = study_set_f.query("age_cat == 1")['mean_IMT'].quantile(0.75)
female_imt_cut_2 = study_set_f.query("age_cat == 2")['mean_IMT'].quantile(0.75)

female_pwv_cut_0 = study_set_f.query("age_cat == 0")['baPWV'].quantile(0.75)
female_pwv_cut_1 = study_set_f.query("age_cat == 1")['baPWV'].quantile(0.75)
female_pwv_cut_2 = study_set_f.query("age_cat == 2")['baPWV'].quantile(0.75)
# %%
study_set_m.loc[(study_set_m['age_cat'] == 0) & (study_set_m['mean_IMT'] >= male_imt_cut_0), "cimt_75"] = 1
study_set_m.loc[(study_set_m['age_cat'] == 1) & (study_set_m['mean_IMT'] >= male_imt_cut_1), "cimt_75"] = 1
study_set_m.loc[(study_set_m['age_cat'] == 2) & (study_set_m['mean_IMT'] >= male_imt_cut_2), "cimt_75"] = 1
study_set_m.loc[study_set_m['cimt_75'].isnull(), "cimt_75"] = 0

study_set_m.loc[(study_set_m['age_cat'] == 0) & (study_set_m['baPWV'] >= male_pwv_cut_0), "bapwv_75"] = 1
study_set_m.loc[(study_set_m['age_cat'] == 1) & (study_set_m['baPWV'] >= male_pwv_cut_1), "bapwv_75"] = 1
study_set_m.loc[(study_set_m['age_cat'] == 2) & (study_set_m['baPWV'] >= male_pwv_cut_2), "bapwv_75"] = 1
study_set_m.loc[study_set_m['bapwv_75'].isnull(), "bapwv_75"] = 0

# %%
study_set_f.loc[(study_set_f['age_cat'] == 0) & (study_set_f['mean_IMT'] >= female_imt_cut_0), "cimt_75"] = 1
study_set_f.loc[(study_set_f['age_cat'] == 1) & (study_set_f['mean_IMT'] >= female_imt_cut_1), "cimt_75"] = 1
study_set_f.loc[(study_set_f['age_cat'] == 2) & (study_set_f['mean_IMT'] >= female_imt_cut_2), "cimt_75"] = 1
study_set_f.loc[study_set_f['cimt_75'].isnull(), "cimt_75"] = 0

study_set_f.loc[(study_set_f['age_cat'] == 0) & (study_set_f['baPWV'] >= female_pwv_cut_0), "bapwv_75"] = 1
study_set_f.loc[(study_set_f['age_cat'] == 1) & (study_set_f['baPWV'] >= female_pwv_cut_1), "bapwv_75"] = 1
study_set_f.loc[(study_set_f['age_cat'] == 2) & (study_set_f['baPWV'] >= female_pwv_cut_2), "bapwv_75"] = 1
study_set_f.loc[study_set_f['bapwv_75'].isnull(), "bapwv_75"] = 0
# %%
study_set = pd.concat((study_set_m, study_set_f), axis=0).reset_index(drop=True)
study_set = study_set.fillna(study_set.median())

study_set.to_csv('../data/study_set_all.csv', index=False, encoding='utf-8')
study_set.query("sex == False").to_csv('../data/study_set_f.csv', index=False, encoding='utf-8')
study_set.query("sex == True").to_csv('../data/study_set_m.csv', index=False, encoding='utf-8')
# %%
