import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer

#Load the data
#Separate time into date and time
df = pd.read_csv('/content/dataset_mood_smartphone.csv', engine= 'python')
df = df.drop('Unnamed: 0',axis=1)
df['Date'] = pd.to_datetime(df['time']).dt.date
df['Time'] = pd.to_datetime(df['time']).dt.time

# aggregate by id and date
res = df.pivot_table(index=[ 'id','Date'], columns='variable',
                     values='value').reset_index()


#replace NAs using KNN
knn_imputer = KNNImputer(n_neighbors=10)
res[['activity', 'appCat.builtin', 'appCat.communication',
       'appCat.entertainment', 'appCat.finance', 'appCat.game',
       'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
       'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call',
       'circumplex.arousal', 'circumplex.valence', 'screen', 'sms']] = knn_imputer.fit_transform(res[['activity', 'appCat.builtin', 'appCat.communication','appCat.entertainment', 'appCat.finance', 'appCat.game',
                                                                                                              'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
                                                                                                              'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call',
                                                                                                              'circumplex.arousal', 'circumplex.valence', 'screen', 'sms']] )
       
res = res.dropna()

# Add day, week, and month
res.Date = pd.to_datetime(res.Date)
res['month'] = [i.month for i in res['Date']]
res['day_of_week'] = [i.dayofweek for i in res['Date']]
res['day_of_year'] = [i.dayofyear for i in res['Date']]

#moving averages, day shifting
res['moving_average_activity'] = res.groupby(['id'])['activity'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_builtin'] = res.groupby(['id'])['appCat.builtin'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_communication'] = res.groupby(['id'])['appCat.communication'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_entertainment'] = res.groupby(['id'])['appCat.entertainment'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_finance'] = res.groupby(['id'])['appCat.finance'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_game'] = res.groupby(['id'])['appCat.game'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_office'] = res.groupby(['id'])['appCat.office'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_other'] = res.groupby(['id'])['appCat.other'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_social'] = res.groupby(['id'])['appCat.social'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_travel'] = res.groupby(['id'])['appCat.travel'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_unknown'] = res.groupby(['id'])['appCat.unknown'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_utilities'] = res.groupby(['id'])['appCat.utilities'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_weather'] = res.groupby(['id'])['appCat.weather'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_call'] = res.groupby(['id'])['call'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_arousal'] = res.groupby(['id'])['circumplex.arousal'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_valence'] = res.groupby(['id'])['circumplex.valence'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_screen'] = res.groupby(['id'])['screen'].rolling(7).mean().reset_index(level=[0,1],drop= True)
res['moving_average_sms'] = res.groupby(['id'])['sms'].rolling(7).mean().reset_index(level=[0,1],drop= True)

# shifts 1 day
res['activity_shift_1d'] = res.groupby('id')['activity'].shift(-1)
res['built_shift_1d'] = res.groupby('id')['appCat.builtin'].shift(-1)
res['communication_shift_1d'] = res.groupby('id')['appCat.communication'].shift(-1)
res['entertainment_shift_1d'] = res.groupby('id')['appCat.entertainment'].shift(-1)
res['finance_shift_1d'] = res.groupby('id')['appCat.finance'].shift(-1)
res['game_shift_1d'] = res.groupby('id')['appCat.game'].shift(-1)
res['office_shift_1d'] = res.groupby('id')['appCat.office'].shift(-1)
res['other_shift_1d'] = res.groupby('id')['appCat.other'].shift(-1)
res['social_shift_1d'] = res.groupby('id')['appCat.social'].shift(-1)
res['travel_shift_1d'] = res.groupby('id')['appCat.travel'].shift(-1)
res['unknown_shift_1d'] = res.groupby('id')['appCat.unknown'].shift(-1)
res['utilities_shift_1d'] = res.groupby('id')['appCat.utilities'].shift(-1)
res['weather_shift_1d'] = res.groupby('id')['appCat.weather'].shift(-1)
res['call_shift_1d'] = res.groupby('id')['call'].shift(-1)
res['arousal_shift_1d'] = res.groupby('id')['circumplex.arousal'].shift(-1)
res['valence_shift_1d'] = res.groupby('id')['circumplex.valence'].shift(-1)
res['screen_shift_1d'] = res.groupby('id')['screen'].shift(-1)
res['sms_shift_1d'] = res.groupby('id')['sms'].shift(-1)

# shifts 2 day
res['activity_shift_2d'] = res.groupby('id')['activity'].shift(-2)
res['built_shift_2d'] = res.groupby('id')['appCat.builtin'].shift(-2)
res['communication_shift_2d'] = res.groupby('id')['appCat.communication'].shift(-2)
res['entertainment_shift_2d'] = res.groupby('id')['appCat.entertainment'].shift(-2)
res['finance_shift_2d'] = res.groupby('id')['appCat.finance'].shift(-2)
res['game_shift_2d'] = res.groupby('id')['appCat.game'].shift(-2)
res['office_shift_2d'] = res.groupby('id')['appCat.office'].shift(-2)
res['other_shift_2d'] = res.groupby('id')['appCat.other'].shift(-2)
res['social_shift_2d'] = res.groupby('id')['appCat.social'].shift(-2)
res['travel_shift_2d'] = res.groupby('id')['appCat.travel'].shift(-2)
res['unknown_shift_2d'] = res.groupby('id')['appCat.unknown'].shift(-2)
res['utilities_shift_2d'] = res.groupby('id')['appCat.utilities'].shift(-2)
res['weather_shift_2d'] = res.groupby('id')['appCat.weather'].shift(-2)
res['call_shift_2d'] = res.groupby('id')['call'].shift(-2)
res['arousal_shift_2d'] = res.groupby('id')['circumplex.arousal'].shift(-2)
res['valence_shift_2d'] = res.groupby('id')['circumplex.valence'].shift(-2)
res['screen_shift_2d'] = res.groupby('id')['screen'].shift(-2)
res['sms_shift_2d'] = res.groupby('id')['sms'].shift(-2)

# exponential moving average: it gives mnore weight to recent events/occurences 
res['ema_activity'] = res.groupby(['id'])['activity'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_builtin'] = res.groupby(['id'])['appCat.builtin'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_communication'] = res.groupby(['id'])['appCat.communication'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_entertainment'] = res.groupby(['id'])['appCat.entertainment'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_finance'] = res.groupby(['id'])['appCat.finance'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_game'] = res.groupby(['id'])['appCat.game'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_office'] = res.groupby(['id'])['appCat.office'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_other'] = res.groupby(['id'])['appCat.other'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_social'] = res.groupby(['id'])['appCat.social'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_travel'] = res.groupby(['id'])['appCat.travel'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_unknown'] = res.groupby(['id'])['appCat.unknown'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_utilities'] = res.groupby(['id'])['appCat.utilities'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_weather'] = res.groupby(['id'])['appCat.weather'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_call'] = res.groupby(['id'])['call'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_arousal'] = res.groupby(['id'])['circumplex.arousal'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_valence'] = res.groupby(['id'])['circumplex.valence'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_screen'] = res.groupby(['id'])['screen'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)
res['ema_sms'] = res.groupby(['id'])['sms'].ewm(span=7, adjust=False).mean().reset_index(level=[0,1],drop= True)

res[['activity', 'appCat.builtin', 'appCat.communication',
       'appCat.entertainment', 'appCat.finance', 'appCat.game',
       'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
       'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call',
       'circumplex.arousal', 'circumplex.valence', 'mood', 'screen', 'sms',
       'month', 'day_of_week', 'day_of_year', 'moving_average_activity',
       'moving_average_builtin', 'moving_average_communication',
       'moving_average_entertainment', 'moving_average_finance',
       'moving_average_game', 'moving_average_office', 'moving_average_other',
       'moving_average_social', 'moving_average_travel',
       'moving_average_unknown', 'moving_average_utilities',
       'moving_average_weather', 'moving_average_call',
       'moving_average_arousal', 'moving_average_valence',
       'moving_average_screen', 'moving_average_sms', 'activity_shift_1d',
       'built_shift_1d', 'communication_shift_1d', 'entertainment_shift_1d',
       'finance_shift_1d', 'game_shift_1d', 'office_shift_1d',
       'other_shift_1d', 'social_shift_1d', 'travel_shift_1d',
       'unknown_shift_1d', 'utilities_shift_1d', 'weather_shift_1d',
       'call_shift_1d', 'arousal_shift_1d', 'valence_shift_1d',
       'screen_shift_1d', 'sms_shift_1d', 'activity_shift_2d',
       'built_shift_2d', 'communication_shift_2d', 'entertainment_shift_2d',
       'finance_shift_2d', 'game_shift_2d', 'office_shift_2d',
       'other_shift_2d', 'social_shift_2d', 'travel_shift_2d',
       'unknown_shift_2d', 'utilities_shift_2d', 'weather_shift_2d',
       'call_shift_2d', 'arousal_shift_2d', 'valence_shift_2d',
       'screen_shift_2d', 'sms_shift_2d', 'ema_activity', 'ema_builtin',
       'ema_communication', 'ema_entertainment', 'ema_finance', 'ema_game',
       'ema_office', 'ema_other', 'ema_social', 'ema_travel', 'ema_unknown',
       'ema_utilities', 'ema_weather', 'ema_call', 'ema_arousal',
       'ema_valence', 'ema_screen', 'ema_sms']] = knn_imputer.fit_transform(res[['activity', 'appCat.builtin', 'appCat.communication',
                                                                                  'appCat.entertainment', 'appCat.finance', 'appCat.game',
                                                                                  'appCat.office', 'appCat.other', 'appCat.social', 'appCat.travel',
                                                                                  'appCat.unknown', 'appCat.utilities', 'appCat.weather', 'call',
                                                                                  'circumplex.arousal', 'circumplex.valence', 'mood', 'screen', 'sms',
                                                                                  'month', 'day_of_week', 'day_of_year', 'moving_average_activity',
                                                                                  'moving_average_builtin', 'moving_average_communication',
                                                                                  'moving_average_entertainment', 'moving_average_finance',
                                                                                  'moving_average_game', 'moving_average_office', 'moving_average_other',
                                                                                  'moving_average_social', 'moving_average_travel',
                                                                                  'moving_average_unknown', 'moving_average_utilities',
                                                                                  'moving_average_weather', 'moving_average_call',
                                                                                  'moving_average_arousal', 'moving_average_valence',
                                                                                  'moving_average_screen', 'moving_average_sms', 'activity_shift_1d',
                                                                                  'built_shift_1d', 'communication_shift_1d', 'entertainment_shift_1d',
                                                                                  'finance_shift_1d', 'game_shift_1d', 'office_shift_1d',
                                                                                  'other_shift_1d', 'social_shift_1d', 'travel_shift_1d',
                                                                                  'unknown_shift_1d', 'utilities_shift_1d', 'weather_shift_1d',
                                                                                  'call_shift_1d', 'arousal_shift_1d', 'valence_shift_1d',
                                                                                  'screen_shift_1d', 'sms_shift_1d', 'activity_shift_2d',
                                                                                  'built_shift_2d', 'communication_shift_2d', 'entertainment_shift_2d',
                                                                                  'finance_shift_2d', 'game_shift_2d', 'office_shift_2d',
                                                                                  'other_shift_2d', 'social_shift_2d', 'travel_shift_2d',
                                                                                  'unknown_shift_2d', 'utilities_shift_2d', 'weather_shift_2d',
                                                                                  'call_shift_2d', 'arousal_shift_2d', 'valence_shift_2d',
                                                                                  'screen_shift_2d', 'sms_shift_2d', 'ema_activity', 'ema_builtin',
                                                                                  'ema_communication', 'ema_entertainment', 'ema_finance', 'ema_game',
                                                                                  'ema_office', 'ema_other', 'ema_social', 'ema_travel', 'ema_unknown',
                                                                                  'ema_utilities', 'ema_weather', 'ema_call', 'ema_arousal',
                                                                                  'ema_valence', 'ema_screen', 'ema_sms']])


#USING RANDOM FOREST REGRESSOR : 10 BEST FEATURES
res_feature_AS14 = res[res['id']== 'AS14.26']

# separate into input and output variables
X = res_feature_AS14.drop(['id', 'mood', 'Date'], axis =1)
y = res_feature_AS14['mood']

# perform feature selection
rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), n_features_to_select=10)
fit = rfe.fit(X, y)

# report selected features
print('Selected Features with RFR:')
names =X.columns.values
for i in range(len(fit.support_)):
	if fit.support_[i]:
		print(names[i])


# USING BORUTA ALGO
# define Boruta feature selection method
model = RandomForestRegressor(n_estimators='auto', max_depth=5, random_state=42)

# find all relevant features
feat_selector = BorutaPy(
    verbose=0,
    estimator=model,
    n_estimators='auto',
    max_iter=500  # number of iterations to perform
)
feat_selector.fit(np.array(X), np.array(y))


# print support and ranking for each feature
print("\n------Support and Ranking for each feature------")
for i in range(len(feat_selector.support_)):
    if feat_selector.support_[i]:
        print("Important Feature : ", X.columns[i],
              " - Ranking: ", feat_selector.ranking_[i])



res_imp_feat = res[['id','Date','appCat.office','circumplex.valence', 'day_of_week', 'travel_shift_1d','game_shift_2d', 'mood']]

res_imp_feat.to_csv('final_dataset.csv',index=False)
  