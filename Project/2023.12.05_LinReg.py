import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


# Prepare dataset
path_to_file='/data/file.csv'
rawdata=pd.read_csv(path_to_file)
# dataset should have indexes of 'frame'
# And columns of 'front_vehicle_position', 'front_vehicle_speed', 'front_vehicle_acceleration', 
# 'back_vehicle_position', 'back_vehicle_speed', 'back_vehicle_acceleration', 
# 'ego_speed', 'ego_acceleration'
# 'ego_speed' is the target variable, accelerations are for later.

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(rawdata, rawdata['ego_speed'], test_size=0.2, random_state=42)

# Standardize dataset
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))

# Plot results
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.show()

# Plot residuals distribution
sns.distplot(residuals)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Plot coefficients
coefficients = pd.DataFrame(model.coef_, rawdata.columns)
coefficients.columns = ['Coefficients']
coefficients.plot(kind='bar')
plt.show()

# Plot cross validation scores
scores = cross_val_score(model, X_train, y_train, cv=5)
plt.plot(scores)
plt.xlabel('Fold')
plt.ylabel('Score')
plt.show()

# Plot learning curve
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5)
plt.plot(train_sizes, train_scores)
plt.plot(train_sizes, test_scores)
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.show()