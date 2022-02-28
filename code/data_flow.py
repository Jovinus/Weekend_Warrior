# %%
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
# %%
df_orig = pd.read_csv("../data/fitness_dataset.csv")
df_orig.head()

print(len(set(df_orig['CDW_NO'])))

print(df_orig.groupby(['CDW_NO'])['sex'].head(1).value_counts())
# %%
tmp = df_orig.query('baPWV.notnull() & mean_IMT.notnull() & CAC.notnull()', engine='python')
# %%
print(len(set(tmp['CDW_NO'])))
print(tmp.groupby(['CDW_NO'])['sex'].head(1).value_counts())
print(tmp.groupby(['CDW_NO','sex']).head(1)['PAP'].value_counts())
print(tmp.groupby(['CDW_NO','sex']).head(1).groupby(['sex'])['PAP'].value_counts())
# %%
tmp = df_orig.query('baPWV.notnull() & mean_IMT.notnull()', engine='python')
# %%
print(len(set(tmp['CDW_NO'])))
print(tmp.groupby(['CDW_NO'])['sex'].head(1).value_counts())
print(tmp.groupby(['CDW_NO','sex']).head(1)['PAP'].value_counts())
print(tmp.groupby(['CDW_NO','sex']).head(1).groupby(['sex'])['PAP'].value_counts())
# %%
