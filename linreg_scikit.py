import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_excel("Baseline_ALL_202303.xlsx", sheet_name="ALL")
#df = pd.read_excel("combined_data.xlsx", sheet_name="Munka1")
#df = pd.read_excel("cymatics_ez_water_experiment_ALL.xlsx", sheet_name="Sheet1")
df = df[~df['Symm1'].isna()]
df['V1Min'].fillna(0, inplace=True) # Replace empty V1min values with 0
df['V1Max'].fillna(df['V1Min'], inplace=True) # Replace empty V1Max values with V1Min

mycolumns = ["Duration [min]", "Humidity [%]", "Air pressure [mb]", "Water temp. [⁰C]", "Air temp. [⁰C]", "Moon illumination", "V1Min", "V1Max"]
X = df[mycolumns].to_numpy()
indices = np.arange(X.shape[0])

##########################################################
################# linreg vs symmetry-fold ################
##########################################################

Y = df["Symm1"].to_numpy()

i_train, i_test, X_train, X_test, Y_train, Y_test = train_test_split(indices, X, Y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred_test = model.predict(X_test)
Y_pred_train = model.predict(X_train)

coefficients = model.coef_
intercept = model.intercept_

print(f'Intercept: {intercept:.2f}')
for icoeff, coeff in enumerate(coefficients):
  print(f'Coefficient {icoeff} for {mycolumns[icoeff]}: {coeff:.2f}')

mse = mean_squared_error(Y_test, Y_pred_test)
print('Mean squared error on test sample:', mse)
mse = mean_squared_error(Y_train, Y_pred_train)
print('Mean squared error on test sample:', mse)

print('Predicted values:', Y_pred_test)
print('Actual values:', Y_test)

print('Predicted values:', Y_pred_train)
print('Actual values:', Y_train)

plt.figure()
plt.title("ML linear regression fit quality check")
plt.ylabel("Symmetry-fold")
plt.xlabel("Measurement #")
plt.plot(i_train, Y_train,      marker='P', color='C0', mfc='none', linestyle='none', label='Measured value, train sample')
plt.plot(i_train, Y_pred_train, marker='o', color='C1', mfc='none', linestyle='none', label='Fit result, train sample')
plt.plot(i_test,  Y_test,       marker='P', color='C0',             linestyle='none', label='Measured value, test sample')
plt.plot(i_test,  Y_pred_test,  marker='o', color='C1',             linestyle='none', label='Fit result, test sample')
plt.legend(loc='upper left')
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.05,box.y0,box.width*1.15,box.height*1.05])
plt.savefig("ML_fit_residuals.png")

##########################################################
################# linreg vs symmetry-fold ################
##########################################################

Y = df["Frequency [Hz]"].to_numpy()

i_train, i_test, X_train, X_test, Y_train, Y_test = train_test_split(indices, X, Y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred_test = model.predict(X_test)
Y_pred_train = model.predict(X_train)

coefficients = model.coef_
intercept = model.intercept_

print(f'Intercept: {intercept:.2f}')
for icoeff, coeff in enumerate(coefficients):
  print(f'Coefficient {icoeff} for {mycolumns[icoeff]}: {coeff:.2f}')

mse = mean_squared_error(Y_test, Y_pred_test)
print('Mean squared error on test sample:', mse)
mse = mean_squared_error(Y_train, Y_pred_train)
print('Mean squared error on train sample:', mse)

print('Predicted values:', Y_pred_test)
print('Actual values:', Y_test)

print('Predicted values:', Y_pred_train)
print('Actual values:', Y_train)

plt.figure()
plt.title("ML linear regression fit quality check")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Measurement #")
plt.plot(i_train, Y_train,      marker='P', color='C0', mfc='none', linestyle='none', label='Measured value, train sample')
plt.plot(i_train, Y_pred_train, marker='o', color='C1', mfc='none', linestyle='none', label='Fit result, train sample')
plt.plot(i_test,  Y_test,       marker='P', color='C0',             linestyle='none', label='Measured value, test sample')
plt.plot(i_test,  Y_pred_test,  marker='o', color='C1',             linestyle='none', label='Fit result, test sample')
plt.legend(loc='upper left')
ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0-box.width*0.05,box.y0,box.width*1.15,box.height*1.05])
plt.savefig("ML_fit_residuals_frequency.png")
