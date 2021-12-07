import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Reading CSV file data
data = pd.read_csv ('data.csv')

#Spliting into training set and test set
training_set, test_set = train_test_split(data, test_size = 0.1, random_state = 2)

x_train = training_set.iloc[:,2:14].values
x_test = test_set.iloc[:,2:14]
y_train = training_set.iloc[:,1].values
y_test = test_set.iloc[:,1]

#fitting training set into linear regression model
regression = linear_model.LinearRegression()
regression.fit(x_train,y_train)

#calculating accuracy of linear regression model using test data
print('Accuracy of Linear Regression Model : ')
print(regression.score(x_test,y_test))


#predicting house prices for x_test using linear regression model
y_predict = regression.predict(x_test)


#making CSV file for combined data
combined_data = pd.concat([pd.DataFrame(x_test),pd.DataFrame(y_test),pd.DataFrame(y_predict)], axis=1, join='inner')
combined_data.rename(columns = {"price": "ActualPrice", 0:"PredictedPrice"}, 
          inplace = True)
combined_data.to_csv('house_price_actual_and_predicted.csv')

#plotting
fig,ax = plt.subplots()
ax.scatter(combined_data.sqft_living, combined_data.ActualPrice,color="red", marker="o")
ax.set_xlabel("Sqft Living",fontsize=14)
ax.set_ylabel("Actual Price",color="red",fontsize=14)
ax2=ax.twinx()
ax2.scatter(combined_data.sqft_living, combined_data["PredictedPrice"],color="blue",marker="o")
ax2.set_ylabel("PredictedPrice",color="blue",fontsize=14)
plt.show()





