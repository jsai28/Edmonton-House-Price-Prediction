import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import math

Edmonton_train = pd.read_csv('preprocessed_train.csv')
Edmonton_test = pd.read_csv('preprocessed_test.csv')
X_train = Edmonton_train.loc[:,['net_area', 'basement_finished', 'has_garage', 'has_fireplace', 'lot_size', 'building_count', 'walkout_basement', 'air_conditioning', 'valuation_group', 'tot_gross_area_description', 'lon', 'lat']]
Y_train = Edmonton_train['assessed_value']

X_test = Edmonton_test.loc[:,['net_area', 'basement_finished', 'has_garage', 'has_fireplace', 'lot_size', 'building_count', 'walkout_basement', 'air_conditioning', 'valuation_group', 'tot_gross_area_description', 'lon', 'lat']]
Y_test = Edmonton_test['assessed_value']

lin_fit = LinearRegression().fit(X_train, Y_train)
lin_pred = lin_fit.predict(X_test)
lin_err = mean_squared_error(Y_test, lin_pred)
lin_r2 = r2_score(Y_test, lin_pred)
lin_mape = mean_absolute_percentage_error(Y_test, lin_pred)

print("model: multi-linear")
print("r2: ", lin_r2)
print("err: ", math.sqrt(lin_err))
print("MAPE: ", lin_mape)

lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(X_train, Y_train)
lasso_predict = lasso.predict(X_test)
lasso_error = mean_squared_error(Y_test, lasso_predict)
lasso_r2 = r2_score(Y_test, lasso_predict)
lasso_mape = mean_absolute_percentage_error(Y_test, lasso_predict)

print(" ")
print("model: lasso")
print("r2: ", lasso_r2)
print("err: ", math.sqrt(lasso_error))
print("MAPE: ", lasso_mape)

ridge = linear_model.Ridge(alpha=1.0)
ridge.fit(X_train, Y_train)
ridge_pred = ridge.predict(X_test)
ridge_err = mean_squared_error(Y_test, ridge_pred)
ridge_r2 = r2_score(Y_test, ridge_pred)
ridge_mape = mean_absolute_percentage_error(Y_test, ridge_pred)

print(" ")
print("model: ridge")
print("r2: ", ridge_r2)
print("err: ", math.sqrt(ridge_err))
print("MAPE: ", ridge_mape)

RFR = RandomForestRegressor(n_estimators=500)
RFR.fit(X_train, Y_train)
RFR_pred = RFR.predict(X_test)
RFR_err = mean_squared_error(Y_test, RFR_pred)
RFR_r2 = r2_score(Y_test, RFR_pred)
RFR_mape = mean_absolute_percentage_error(Y_test, RFR_pred)

print(" ")
print("model: RFR")
print("r2: ", RFR_r2)
print("err: ", math.sqrt(RFR_err))
print("MAPE: ", RFR_mape)

xgb = XGBRegressor(n_estimators=500)
xgb.fit(X_train, Y_train, verbose=False)
xgb_pred = xgb.predict(X_test)
xgb_err = mean_squared_error(Y_test, xgb_pred)
xgb_r2 = r2_score(Y_test, xgb_pred)
xgb_mape = mean_absolute_percentage_error(Y_test, xgb_pred)

print(" ")
print("model: xgboost")
print("r2: ", xgb_r2)
print("err: ", math.sqrt(xgb_err))
print("MAPE: ", xgb_mape)

combined = np.array([lin_pred, lasso_predict, ridge_pred, RFR_pred, xgb_pred])
results = pd.DataFrame({
    'True Assessed Value': Y_test,
    'multi-linear': lin_pred.tolist(),
    'lasso': lasso_predict.tolist(),
    'ridge': ridge_pred.tolist(),
    'RFR': RFR_pred.tolist(),
    'XGBoost': xgb_pred.tolist(),
})
results.to_csv("results.csv")