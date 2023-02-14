import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer




#----------------------------------------------------------------------------------------------------------------
data_file_frame  = pd.read_csv('Car_details.csv', sep=',')
data_file_frame.head()

data_file_frame = data_file_frame[['name','year','km_driven', 'fuel', 'seller_type', 'transmission',
       'owner', 'mileage', 'engine', 'max_power','torque' ,'seats','selling_price']]

#--------------------------------------------------------------------------------------------------------------
data_file_frame.drop('year', axis=1, inplace=True)
data_file_frame.drop('name', axis=1, inplace=True)
data_file_frame.drop('torque', axis=1, inplace=True)

#-------------------------------------------------------------------------------------------------------------
data_file_frame['mileage'] = data_file_frame['mileage'].str.replace(r'kmpl', '')
data_file_frame['engine'] = data_file_frame['engine'].str.replace(r'CC', '')
data_file_frame['max_power'] = data_file_frame['max_power'].str.replace(r'bhp', '')

#-----------------------------------------------------------------------------------------------------------
data_file_frame.isnull().sum()


#---------------------------------------------------------------------------------------------------------
data_file_frame = data_file_frame[~data_file_frame.mileage.isnull()].copy()
data_file_frame.isnull().sum()

#-------------------------------------------------------------------------------------------------------
data_file_frame.dtypes

#-------------------------------------------------------------------------------------------

data_file_frame['owner'] = data_file_frame.owner.astype('category')
data_file_frame['transmission'] = data_file_frame.transmission.astype('category')
data_file_frame['fuel'] = data_file_frame.fuel.astype('category')
data_file_frame['seller_type'] = data_file_frame.seller_type.astype('category')


data_file_frame['owner'] = data_file_frame['owner'].cat.codes.astype('category')
data_file_frame['transmission'] = data_file_frame['transmission'].cat.codes.astype('category')
data_file_frame['fuel'] = data_file_frame['fuel'].cat.codes.astype('category')
data_file_frame['seller_type'] = data_file_frame['seller_type'].cat.codes.astype('category')


data_file_frame.dtypes

data_file_frame.isnull().sum()


data_file_frame.transmission.value_counts(normalize=True)

#now converting the above into the respective plot
data_file_frame.transmission.value_counts(normalize=True).plot.barh()
plt.show()


data_file_frame.owner.value_counts(normalize=True)

#now converting the above into the respective plot
data_file_frame.owner.value_counts(normalize=True).plot.barh()
plt.show()


data_file_frame.fuel.value_counts(normalize=True)

#now converting the above into the respective plot
data_file_frame.fuel.value_counts(normalize=True).plot.barh()
plt.show()


data_file_frame.seller_type.value_counts(normalize=True)

#now converting the above into the respective plot
data_file_frame.seller_type.value_counts(normalize=True).plot.pie()
plt.show()


sns.pairplot(data = data_file_frame, vars=['selling_price','km_driven'])
plt.show()


data_file_frame[['selling_price', 'km_driven']].corr()

#plot the correlation matrix of salary, balance and age in data dataframe.
sns.heatmap(data_file_frame[['selling_price', 'km_driven']].corr(), annot=True, cmap = 'Reds')
plt.show()


data_file_frame.seats.describe()

data_file_frame.max_power = pd.to_numeric(data_file_frame.max_power, errors='coerce')


data_file_frame.mileage = pd.to_numeric(data_file_frame.mileage, errors='coerce')

data_file_frame['mileage'] = data_file_frame['mileage'].multiply(100) 

data_file_frame.engine = pd.to_numeric(data_file_frame.engine, errors='coerce')

X = data_file_frame.iloc[:, :-1].values

y = data_file_frame.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train = np.nan_to_num(X_train)

y_train = np.nan_to_num(y_train)

# Fitting Multiple Linear Regression to the Training set
regression = LinearRegression()
regression.fit(X_train, y_train)

X_test = np.nan_to_num(X_test)

y_pred = regression.predict(X_test)

df = pd.DataFrame(data=y_test, columns=['y_test'])
df['y_pred'] = y_pred

a = [126000,1,1,2,1,21,1410,100,5]
b = np.array(a)
b = b.reshape(1, -1)
y_pred_single_obs = regression.predict(b)
round(float(y_pred_single_obs), 2)

r2_score(y_test, y_pred)

import joblib
joblib.dump(regression, "flask_model.pkl")

driven = 122000
fuel = 1
seller_type = 0
transmission = 1
owner = 1
mileage = 150
engine = 1400
max_power = 100
seats = 5

y_predictors = [driven,fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]
predictors_array = np.array(y_predictors)
pred_args_arr = predictors_array.reshape(1, -1)

regression_model = open('flask_model.pkl', 'rb')
machine_learning_model = joblib.load(regression_model)
model_prediction = machine_learning_model.predict(pred_args_arr)

print(round(float(model_prediction),2))
