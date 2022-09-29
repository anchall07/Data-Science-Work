View
Insert
Cell
Kernel
Widgets
Help

Code
import numpy as np

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as ts
from sklearn.model_selection import train_test_split
df = pd.read_csv('kc_house_data.csv')
print(df.columns.values)

df.info()

df.describe().transpose()

sn.set(style="whitegrid", font_scale=1)
plt.figure(figsize=(13,13))
plt.title('Correlation Matrix',fontsize=25)
sn.heatmap(df.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="GnBu",linecolor='w',
            annot=True, annot_kws={"size":7}, cbar_kws={"shrink": .7})

price_corr = df.corr()['price'].sort_values(ascending=False)

f, axes = plt.subplots(1, 2,figsize=(15,5))
sn.histplot(df['price'], ax=axes[0])
sn.scatterplot(x='price',y='sqft_living', data=df, ax=axes[1])
sn.despine(bottom=True, left=True)
axes[0].set(xlabel='Price in millions [USD]', ylabel='', title='Price Distribuition')
axes[1].set(xlabel='Price', ylabel='Sqft Living', title='Price vs Sqft Living')
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()

sn.set(style="whitegrid", font_scale=1)
f, axes = plt.subplots(1, 2,figsize=(15,5))
sn.boxplot(x=df['bedrooms'],y=df['price'], ax=axes[0])
sn.boxplot(x=df['floors'],y=df['price'], ax=axes[1])
sn.despine(bottom=True, left=True)
axes[0].set(xlabel='Bedrooms', ylabel='Price', title='Bedrooms vs Price Box Plot')
axes[1].set(xlabel='Floors', ylabel='Price', title='Floors vs Price Box Plot')
[Text(0.5, 0, 'Floors'),
 Text(0, 0.5, 'Price'),
 Text(0.5, 1.0, 'Floors vs Price Box Plot')]

f, axes = plt.subplots(1, 2,figsize=(15,5))
sn.boxplot(x=df['waterfront'],y=df['price'], ax=axes[0])
sn.boxplot(x=df['view'],y=df['price'], ax=axes[1])
sn.despine(left=True, bottom=True)
axes[0].set(xlabel='Waterfront', ylabel='Price', title='Waterfront vs Price Box Plot')
axes[1].set(xlabel='View', ylabel='Price', title='View vs Price Box Plot')

f, axe = plt.subplots(1, 1,figsize=(15,5))
sn.boxplot(x=df['grade'],y=df['price'], ax=axe)
sn.despine(left=True, bottom=True)
axe.set(xlabel='Grade', ylabel='Price', title='Grade vs Price Box Plot')
[Text(0.5, 0, 'Grade'),
 Text(0, 0.5, 'Price'),
 Text(0.5, 1.0, 'Grade vs Price Box Plot')]

df = df.drop('id', axis=1)
df = df.drop('zipcode',axis=1)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)
df = df.drop('date',axis=1)


f, axes = plt.subplots(1, 2,figsize=(15,5))
sn.boxplot(x='year',y='price',data=df, ax=axes[0])
sn.boxplot(x='month',y='price',data=df, ax=axes[1])
sn.despine(left=True, bottom=True)
axes[0].set(xlabel='Year', ylabel='Price', title='Price by Year Box Plot')
axes[1].set(xlabel='Month', ylabel='Price', title='Price by Month Box Plot')

f, axe = plt.subplots(1, 1,figsize=(15,5))
df.groupby('month').mean()['price'].plot()
sn.despine(left=True, bottom=True)
axe.set(xlabel='Month', ylabel='Price', title='Price Trends')
[Text(0.5, 0, 'Month'), Text(0, 0.5, 'Price'), Text(0.5, 1.0, 'Price Trends')]


X = df.drop('price',axis=1)

y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.preprocessing import MinMaxScaler
# creating a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

# evaluation on test data
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
from sklearn.metrics import classification_report,confusion_matrix
scaler = MinMaxScaler()

# fit and transfrom
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# everything has been scaled between 1 and 0
print('Max: ',X_train.max())
print('Min: ', X_train.min())

model = Sequential()
# input layer
model.add(Dense(19,activation='relu'))
# hidden layers
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
# output layer
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)

losses = pd.DataFrame(model.history.history)
plt.figure(figsize=(15,5))
sn.lineplot(data=losses,lw=3)
plt.xlabel('Epochs')
plt.ylabel('')
plt.title('Training Loss per Epoch')
sn.despine()


predictions = model.predict(X_test)
print('MAE: ',mean_absolute_error(y_test,predictions))
print('MSE: ',mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,predictions)))
print('Variance Regression Score: ',explained_variance_score(y_test,predictions))

print('\n\nDescriptive Statistics:\n',df['price'].describe())


f, axes = plt.subplots(1, 2,figsize=(15,5))
plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')

errors = y_test.values.reshape(6484, 1) - predictions
sn.histplot(errors, ax=axes[0])

sn.despine(left=True, bottom=True)
axes[0].set(xlabel='Error', ylabel='', title='Error Histogram')
axes[1].set(xlabel='Test True Y', ylabel='Model Predictions', title='Model Predictions vs Perfect Fit')

[Text(0.5, 0, 'Test True Y'),
 Text(0, 0.5, 'Model Predictions'),
 Text(0.5, 1.0, 'Model Predictions vs Perfect Fit')]

single_house = df.drop('price',axis=1).iloc[0]
print(f'Features of new house:\n{single_house}')
single_house = scaler.transform(single_house.values.reshape(-1, 19))
print('\nPrediction Price:',model.predict(single_house)[0,0])
print('\nOriginal Price:',df.iloc[0]['price'])
