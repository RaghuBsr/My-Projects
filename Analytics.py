import pandas as pd
import seaborn as sns
sns.set_context(font_scale=2)
#Load the data files
root = 'C:/New folder/'
df_train=pd.read_excel(root+'Adops & Data Scientist Sample Data.xlsx')
#************************************************************************************************
#Question 1
df_bdvr = df_train.loc[df_train['country_id'] == 'BDV']
unique_users = df_bdvr.groupby('site_id')['user_id'].agg(['unique'])
df_site=unique_users["unique"].apply(lambda x: len(x))
print("Site ID with Larget unique user count in BDV :",unique_users.index[1],"with",df_site[1],"users.")
#************************************************************************************************
#Question 2:
df_train.info()
df_train["ts"]=pd.to_datetime(df_train["ts"])
df_days = df_train[(df_train['ts'] >= '2019-02-03') & (df_train['ts'] < '2019-02-05')]
df_morethan10=df_days.groupby('user_id').filter(lambda x: len(x) > 10)
df_daysgrouped=df_morethan10.groupby(['user_id','site_id'],sort=False)['user_id'].agg(['count']).reset_index()
df_daysgroupedF=df_daysgrouped.sort_values('user_id', ascending=True).drop_duplicates('user_id', keep='first')
df_fourusers=df_daysgroupedF.sort_values('count',ascending=False)
print(df_fourusers)
#************************************************************************************************
#question 3
df_lasttime=df_train.groupby(['user_id'],sort=False)['ts'].agg(['max']).reset_index()
new_df = pd.merge(df_lasttime, df_train,  how='left', left_on=['user_id','max'], right_on = ['user_id','ts'])
df_siteId=new_df.groupby(['site_id'],sort=False)['user_id'].agg(['count'])
df_threepairs=df_siteId.sort_values('count',ascending=False).head(3)
print(df_threepairs)
#************************************************************************************************
#question 4
df_last=df_train.groupby(['user_id'],sort=False)['ts'].agg(['max']).reset_index()
df_first=df_train.groupby(['user_id'],sort=False)['ts'].agg(['min']).reset_index()
df_lastsite = pd.merge(df_last, df_train,  how='left', left_on=['user_id','max'], right_on = ['user_id','ts'])
df_firstsite = pd.merge(df_first, df_train,  how='left', left_on=['user_id','min'], right_on = ['user_id','ts'])
finalUsersite=pd.merge(df_lastsite, df_firstsite,  how='inner', left_on=['user_id','site_id'], right_on = ['user_id','site_id'])
df_Totnumber=len(finalUsersite.groupby(['user_id'])['user_id'].agg(['count']))
print(df_Totnumber)
#************************************************************************************************    











