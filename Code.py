## Developing the model ###

# Load Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from numpy import where

############################################### Regression and Classification Tasks ###############################3####

########################## Import data ##########################################
kickstarter_df = pd.read_excel(r"D:\OneDrive\Desktop\INSY662\Individual Project\Kickstarter.xlsx")

df = kickstarter_df.drop(labels=['project_id','name','disable_communication',
                                 'staff_pick','backers_count','spotlight','state_changed_at',
                                 'state_changed_at_weekday','state_changed_at_month',
                                 'state_changed_at_day','state_changed_at_yr',
                                 'state_changed_at_hr','launch_to_state_change_days'],axis=1)


df = df[(df['state'] == 'successful')|(df['state'] == 'failed')]
df['category'] = df['category'].fillna('missing')
df = df.dropna()

########################## Removing the anomalies ##############################

df1 = df.drop(labels=['deadline','created_at','launched_at'],axis=1)
df1 = pd.get_dummies(df1,columns=['state','country','currency','category',
                                  'deadline_weekday','created_at_weekday','launched_at_weekday'])


iforest = IsolationForest(n_estimators=100, contamination=0.02,random_state=3)
pred = iforest.fit_predict(df1)
score = iforest.decision_function(df1)
anom_index = where(pred==-1)
values = df.iloc[anom_index]

df=pd.concat([df,values]).drop_duplicates(keep=False)

########################## Setting up the independent variables ##############

y_reg = df['usd_pledged']

df = df.assign(outcome = (df.state=='successful'))
df['outcome']=df['outcome'].astype(int)
y_class = df['outcome']


######################### Pre-processing for regression and classification models###

df = df.assign(usd_goal=df.goal*df.static_usd_rate)
df = df.drop(axis=1,labels=['outcome','usd_pledged','state','goal','static_usd_rate','pledged','currency'])

df['deadline'] = pd.to_datetime(df['deadline'])
df['created_at'] = pd.to_datetime(df['created_at'])
df['launched_at'] = pd.to_datetime(df['launched_at'])

base_deadline_day = min(df['deadline'])
base_created_at_day = min(df['created_at'])
base_launched_at_day = min(df['launched_at'])

df = df.assign(days_since_base_deadline=df.deadline-base_deadline_day)
df['days_since_base_deadline'] = df['days_since_base_deadline'].astype('timedelta64[D]')
df = df.assign(days_since_base_created_at=df.created_at-base_created_at_day)
df['days_since_base_created_at'] = df['days_since_base_created_at'].astype('timedelta64[D]')
df = df.assign(days_since_base_launched_at=df.launched_at-base_launched_at_day)
df['days_since_base_launched_at'] = df['days_since_base_launched_at'].astype('timedelta64[D]')

df['deadline_dayofweek'] = df['deadline'].dt.dayofweek
df['created_at_dayofweek'] = df['created_at'].dt.dayofweek
df['launched_at_dayofweek'] = df['launched_at'].dt.dayofweek

df = df.drop(labels=['deadline','created_at','launched_at'],axis=1)

df2 = df.copy()

df2.loc[(df2.category!='Sound')&
       (df2.category!='missing')&
       (df2.category!='Hardware')&
       (df2.category!='Wearables')&
       (df2.category!='Gadgets')&
       (df2.category!='Web')&
       (df2.category!='Software')&
       (df2.category!='Robots')&
       (df2.category!='Makerspace')&
       (df2.category!='Flight')&
       (df2.category!='Apps')&
       (df2.category!='Plays')
       ,'category'] = 'Other'

df2.loc[(df2.country!='US')&
       (df2.country!='GB')&
       (df2.country!='DE')&
       (df2.country!='CA')&
       (df2.country!='NL')&
       (df2.country!='AU')&
       (df2.country!='SE')&
       (df2.country!='ES')
       ,'country'] = 'Other_country'

df2.loc[(df2.created_at_weekday!='Tuesday')&
       (df2.created_at_weekday!='Wednesday')&
       (df2.created_at_weekday!='Friday')&
       (df2.created_at_weekday!='Monday')&
       (df2.created_at_weekday!='Thursday')
       ,'created_at_weekday'] = 'created_at_Weekend'

df2.loc[(df2.created_at_weekday!='created_at_Weekend'),'created_at_weekday']='created_at_Weekday'

df2.loc[(df2.launched_at_weekday!='Tuesday')&
       (df2.launched_at_weekday!='Wednesday')&
       (df2.launched_at_weekday!='Friday')&
       (df2.launched_at_weekday!='Monday')&
       (df2.launched_at_weekday!='Thursday')
       ,'launched_at_weekday'] = 'launched_at_Weekend'

df2.loc[(df2.launched_at_weekday!='launched_at_Weekend'),'launched_at_weekday']='launched_at_Weekday'

df2.loc[(df2.deadline_weekday!='Tuesday')&
       (df2.deadline_weekday!='Wednesday')&
       (df2.deadline_weekday!='Friday')&
       (df2.deadline_weekday!='Monday')&
       (df2.deadline_weekday!='Thursday')
       ,'deadline_weekday'] = 'deadline_Weekend'

df2.loc[(df2.deadline_weekday!='deadline_Weekend'),'deadline_weekday']='deadline_Weekday'

df_continuous = pd.get_dummies(df2,columns=['category','country','created_at_weekday',
                                                      'launched_at_weekday','deadline_weekday'])

X_final = df_continuous.drop(labels=['name_len',
                                     'created_at_yr',
                                     'launched_at_yr',
                                     'deadline_yr',
                                     'days_since_base_deadline',
                                     'days_since_base_created_at'],axis=1)

########################### Buildeing the models #################################

## Regression Model - Random Forest Regressor

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_final, y_reg, test_size = 0.3, random_state = 4)

# Build the model
rf = RandomForestRegressor(random_state=0,n_estimators=100,max_features=7)
model1 = rf.fit(X_train, y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = model1.predict(X_test)

# Calculate the mean squared error of the prediction
mean_squared_error(y_test, y_test_pred)


## Classification Model - Gradient Boosting Classifier

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_final, y_class, test_size = 0.3, random_state = 4)

# Build the model
gbt = GradientBoostingClassifier(n_estimators=100,min_samples_split=3)
model2 = gbt.fit(X_train, y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = model2.predict(X_test)

# Calculate the mean squared error of the prediction
accuracy_score(y_test, y_test_pred)
f1_score(y_test, y_test_pred)



############################ Grading for Regression and Classification #############


############################ Import Grading dataset ##############################
kickstarter_grading_df = pd.read_excel(r"D:\OneDrive\Desktop\INSY662\Individual Project\Kickstarter-Grading-Sample.xlsx")

df = kickstarter_grading_df.drop(labels=['project_id','name','disable_communication',
                                 'staff_pick','backers_count','spotlight','state_changed_at',
                                 'state_changed_at_weekday','state_changed_at_month',
                                 'state_changed_at_day','state_changed_at_yr',
                                 'state_changed_at_hr','launch_to_state_change_days'],axis=1)

############################ Pre-processing of the grading data ##################

df = df[(df['state'] == 'successful')|(df['state'] == 'failed')]
df['category'] = df['category'].fillna('missing')


df = df.dropna()

df = df.assign(outcome = (df.state=='successful'))
df['outcome']=df['outcome'].astype(int)
y_grading_class = df['outcome']

y_grading_reg = df['usd_pledged']


df = df.assign(usd_goal=df.goal*df.static_usd_rate)
df = df.drop(axis=1,labels=['outcome','usd_pledged','state','goal','static_usd_rate','pledged'])

df['deadline'] = pd.to_datetime(df['deadline'])
df['created_at'] = pd.to_datetime(df['created_at'])
df['launched_at'] = pd.to_datetime(df['launched_at'])

base_deadline_day = min(df['deadline'])
base_created_at_day = min(df['created_at'])
base_launched_at_day = min(df['launched_at'])

df = df.assign(days_since_base_deadline=df.deadline-base_deadline_day)
df['days_since_base_deadline'] = df['days_since_base_deadline'].astype('timedelta64[D]')
df = df.assign(days_since_base_created_at=df.created_at-base_created_at_day)
df['days_since_base_created_at'] = df['days_since_base_created_at'].astype('timedelta64[D]')
df = df.assign(days_since_base_launched_at=df.launched_at-base_launched_at_day)
df['days_since_base_launched_at'] = df['days_since_base_launched_at'].astype('timedelta64[D]')

df['deadline_dayofweek'] = df['deadline'].dt.dayofweek
df['created_at_dayofweek'] = df['created_at'].dt.dayofweek
df['launched_at_dayofweek'] = df['launched_at'].dt.dayofweek

df = df.drop(labels=['deadline','created_at','launched_at'],axis=1)


df2 = df.copy()

df2.loc[(df2.category!='Sound')&
       (df2.category!='missing')&
       (df2.category!='Hardware')&
       (df2.category!='Wearables')&
       (df2.category!='Gadgets')&
       (df2.category!='Web')&
       (df2.category!='Software')&
       (df2.category!='Robots')&
       (df2.category!='Makerspace')&
       (df2.category!='Flight')&
       (df2.category!='Apps')&
       (df2.category!='Plays')
       ,'category'] = 'Other'

df2.loc[(df2.country!='US')&
       (df2.country!='GB')&
       (df2.country!='DE')&
       (df2.country!='CA')&
       (df2.country!='NL')&
       (df2.country!='AU')&
       (df2.country!='SE')&
       (df2.country!='ES')
       ,'country'] = 'Other_country'

df2.loc[(df2.created_at_weekday!='Tuesday')&
       (df2.created_at_weekday!='Wednesday')&
       (df2.created_at_weekday!='Friday')&
       (df2.created_at_weekday!='Monday')&
       (df2.created_at_weekday!='Thursday')
       ,'created_at_weekday'] = 'created_at_Weekend'

df2.loc[(df2.created_at_weekday!='created_at_Weekend'),'created_at_weekday']='created_at_Weekday'

df2.loc[(df2.launched_at_weekday!='Tuesday')&
       (df2.launched_at_weekday!='Wednesday')&
       (df2.launched_at_weekday!='Friday')&
       (df2.launched_at_weekday!='Monday')&
       (df2.launched_at_weekday!='Thursday')
       ,'launched_at_weekday'] = 'launched_at_Weekend'

df2.loc[(df2.launched_at_weekday!='launched_at_Weekend'),'launched_at_weekday']='launched_at_Weekday'

df2.loc[(df2.deadline_weekday!='Tuesday')&
       (df2.deadline_weekday!='Wednesday')&
       (df2.deadline_weekday!='Friday')&
       (df2.deadline_weekday!='Monday')&
       (df2.deadline_weekday!='Thursday')
       ,'deadline_weekday'] = 'deadline_Weekend'

df2.loc[(df2.deadline_weekday!='deadline_Weekend'),'deadline_weekday']='deadline_Weekday'


df_continuous = df2.drop(labels=['currency'],axis=1)



df_continuous = pd.get_dummies(df_continuous,columns=['category','country','created_at_weekday',
                                                      'launched_at_weekday','deadline_weekday'])

X_grading = df_continuous.drop(labels=['name_len',
                                     'created_at_yr',
                                     'launched_at_yr',
                                     'deadline_yr',
                                     'days_since_base_deadline',
                                     'days_since_base_created_at'],axis=1)



#################### Predicting using the trained models #################



## Regression Task

y_grading_reg_pred = model1.predict(X_grading)

mean_squared_error(y_grading_reg, y_grading_reg_pred)

## Classification Task

y_grading_class_pred = model2.predict(X_grading)

accuracy_score(y_grading_class, y_grading_class_pred)
f1_score(y_grading_class, y_grading_class_pred)

#####################################################################################################

#################################################### Clustering Model ####################################

########################## Import data ##########################################

df = kickstarter_df.drop(labels=['project_id','name','disable_communication',
                                 'staff_pick','backers_count','spotlight','state_changed_at',
                                 'state_changed_at_weekday','state_changed_at_month',
                                 'state_changed_at_day','state_changed_at_yr',
                                 'state_changed_at_hr','launch_to_state_change_days'],axis=1)


df = df[(df['state'] == 'successful')|(df['state'] == 'failed')]
df['category'] = df['category'].fillna('missing')
df = df.dropna()

df1 = df.drop(labels=['deadline','created_at','launched_at'],axis=1)
df1 = pd.get_dummies(df1,columns=['state','country','currency','category',
                                  'deadline_weekday','created_at_weekday','launched_at_weekday'])


########################## Removing the anomalies ##############################


iforest = IsolationForest(n_estimators=100, contamination=0.02,random_state=3)
pred = iforest.fit_predict(df1)
score = iforest.decision_function(df1)
anom_index = where(pred==-1)
values = df.iloc[anom_index]

df=pd.concat([df,values]).drop_duplicates(keep=False)

######################### Clustering model ######################


df = df.assign(outcome = (df.state=='successful'))
df['outcome']=df['outcome'].astype(int)
y_class = df['outcome']


df = df.assign(usd_goal=df.goal*df.static_usd_rate)
df = df.drop(axis=1,labels=['state','goal','static_usd_rate','pledged'])

df['deadline'] = pd.to_datetime(df['deadline'])
df['created_at'] = pd.to_datetime(df['created_at'])
df['launched_at'] = pd.to_datetime(df['launched_at'])

base_deadline_day = min(df['deadline'])
base_created_at_day = min(df['created_at'])
base_launched_at_day = min(df['launched_at'])

df = df.assign(days_since_base_deadline=df.deadline-base_deadline_day)
df['days_since_base_deadline'] = df['days_since_base_deadline'].astype('timedelta64[D]')
df = df.assign(days_since_base_created_at=df.created_at-base_created_at_day)
df['days_since_base_created_at'] = df['days_since_base_created_at'].astype('timedelta64[D]')
df = df.assign(days_since_base_launched_at=df.launched_at-base_launched_at_day)
df['days_since_base_launched_at'] = df['days_since_base_launched_at'].astype('timedelta64[D]')

df['deadline_dayofweek'] = df['deadline'].dt.dayofweek
df['created_at_dayofweek'] = df['created_at'].dt.dayofweek
df['launched_at_dayofweek'] = df['launched_at'].dt.dayofweek

df = df.drop(labels=['deadline','created_at','launched_at'],axis=1)

df1 = df.copy()

df1=df.copy()
df1.loc[(df1.category!='Sound')&
       (df1.category!='missing')&
       (df1.category!='Hardware')&
       (df1.category!='Wearables')&
       (df1.category!='Gadgets')&
       (df1.category!='Web')&
       (df1.category!='Software')&
       (df1.category!='Robots')&
       (df1.category!='Makerspace')&
       (df1.category!='Flight')&
       (df1.category!='Apps')&
       (df1.category!='Plays')
       ,'category'] = 'Other'

df1=df.copy()
df1.loc[(df1.category!='Sound')&
       (df1.category!='Hardware')&
       (df1.category!='Wearables')&
       (df1.category!='Gadgets')
       ,'category'] = 'Other'

df1.loc[(df1.country!='US')
       ,'country'] = 'Other_country'

df1.loc[(df1.created_at_weekday!='Tuesday')&
       (df1.created_at_weekday!='Wednesday')&
       (df1.created_at_weekday!='Friday')&
       (df1.created_at_weekday!='Monday')&
       (df1.created_at_weekday!='Thursday')
       ,'created_at_weekday'] = 'created_at_Weekend'

df1.loc[(df1.created_at_weekday!='created_at_Weekend'),'created_at_weekday']='created_at_Weekday'

df1.loc[(df1.launched_at_weekday!='Tuesday')&
       (df1.launched_at_weekday!='Wednesday')&
       (df1.launched_at_weekday!='Friday')&
       (df1.launched_at_weekday!='Monday')&
       (df1.launched_at_weekday!='Thursday')
       ,'launched_at_weekday'] = 'launched_at_Weekend'

df1.loc[(df1.launched_at_weekday!='launched_at_Weekend'),'launched_at_weekday']='launched_at_Weekday'

df1.loc[(df1.deadline_weekday!='Tuesday')&
       (df1.deadline_weekday!='Wednesday')&
       (df1.deadline_weekday!='Friday')&
       (df1.deadline_weekday!='Monday')&
       (df1.deadline_weekday!='Thursday')
       ,'deadline_weekday'] = 'deadline_Weekend'

df1.loc[(df1.deadline_weekday!='deadline_Weekend'),'deadline_weekday']='deadline_Weekday'


df_continuous = df1.drop(labels=['currency'],axis=1)



X_final = pd.get_dummies(df_continuous,columns=['category','country','created_at_weekday',
                                                      'launched_at_weekday','deadline_weekday'])

################# Setting the variables for clustering ##########################

X_cluster = X_final[['usd_goal','outcome','days_since_base_launched_at',
                     'create_to_launch_days','launch_to_deadline_days',
                     'category_Sound','category_Hardware',
                     'category_Wearables','category_Gadgets',
                     'category_Other','name_len_clean','blurb_len_clean']]


## Standardization

scaler = MinMaxScaler()
X_std = scaler.fit_transform(X_cluster)

########################### Buildeing the models #################################


   
kmeans = KMeans(n_clusters=8)
model = kmeans.fit(X_std)
labels = model.labels_    

silhouette = silhouette_samples(X_std,labels)

## Silhouette Score for each clusters

temp = pd.DataFrame({'label':labels,'silhouette':silhouette})
print('Average Silhouette Score for Cluster 1: ',np.average(temp[temp['label'] == 0].silhouette))
print('Average Silhouette Score for Cluster 2: ',np.average(temp[temp['label'] == 1].silhouette))
print('Average Silhouette Score for Cluster 3: ',np.average(temp[temp['label'] == 2].silhouette))
print('Average Silhouette Score for Cluster 4: ',np.average(temp[temp['label'] == 3].silhouette))
print('Average Silhouette Score for Cluster 5: ',np.average(temp[temp['label'] == 4].silhouette))
print('Average Silhouette Score for Cluster 6: ',np.average(temp[temp['label'] == 5].silhouette))
print('Average Silhouette Score for Cluster 7: ',np.average(temp[temp['label'] == 6].silhouette))
print('Average Silhouette Score for Cluster 8: ',np.average(temp[temp['label'] == 7].silhouette))


## Average Silhouette score for the model

print('Silluette Score : ',silhouette_score(X_std,model.labels_))
centers = pd.DataFrame(model.cluster_centers_, columns = X_cluster.columns)






















