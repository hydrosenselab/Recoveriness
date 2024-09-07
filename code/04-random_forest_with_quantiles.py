"""
Program to predict the recovery data
using the trained Quantile Regression
Forest model.
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from quantile_forest import RandomForestQuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score



df_flashiness = pd.read_csv(filepath_or_buffer="../data/flashiness.csv", sep=",",
                            index_col=None, header=0)
df_ = pd.read_csv(filepath_or_buffer="data_prepared.csv", sep=",",
                  index_col=0, header=0)
df_prediction = pd.read_csv(filepath_or_buffer="../data/ungauged_.csv", sep=",",
                            index_col=None, header=0).drop(labels=["row", "col"], axis=1)
df_test = df_prediction.copy(deep=False)

df_prediction.columns = df_.columns[:-1]
df_test.columns = df_.columns[:-1]

df_test.insert(
    loc=0,
    column='lat',
    value=df_flashiness["lat"])
df_test.insert(
    loc=1,
    column='lon',
    value=df_flashiness["lon"])

"""
Train-Test Split
"""
X, y = df_.drop(labels="mr.ecdf", axis=1), df_[["mr.ecdf"]]
Xtrain_, Xtest_, ytrain_, ytest_ = train_test_split(X, y, test_size=0.20, random_state=21)
ytrain_, ytest_ = np.array(ytrain_).squeeze(), np.array(ytest_).squeeze()
print()

"""
Hyperparameters
"""
ntrees = np.arange(10, 11, 1)
maxdepth = np.arange(7, 8, 2)
maxfeatures = np.arange(14, 15, 1)

naccu = len(ntrees) * len(maxdepth) * len(maxfeatures)

rfr_ = list()
qfr_ = list()
gbr_ = list()
param_ = list()
counter = 1
for i, nt in enumerate(ntrees):
    for j, md in enumerate(maxdepth):
        for k, mf in enumerate(maxfeatures):
            param_.append([nt, md, mf])
            print([nt, md, mf], "\t\t\t\t\t\t\t\t: {}/{}".format(counter, naccu))

            """
            Random Forest Regressor
            """
            rfr = RandomForestRegressor(n_estimators=nt, max_depth=md, max_features=mf)
            rfr.fit(Xtrain_, ytrain_)
            ypred_rfr = rfr.predict(Xtest_)
            accu_rfr = r2_score(y_true=ytest_, y_pred=ypred_rfr)
            print("Random Forest accuracy: {:.2f}".format(accu_rfr))
            rfr_.append(accu_rfr)

            """
            Quantile Forest Regression
            """
            qfr = RandomForestQuantileRegressor(n_estimators=nt, max_depth=md, max_features=mf)
            qfr.fit(Xtrain_, ytrain_)
            ypred_qfr = qfr.predict(Xtest_)
            accu_qfr = r2_score(y_true=ytest_, y_pred=ypred_qfr)
            print("Quantile Random Forest accuracy: {:.2f}".format(accu_qfr))
            qfr_.append(accu_qfr)

            """
            Gradient Boosting Regressor
            """
            gbr = GradientBoostingRegressor(n_estimators=nt, max_depth=md, max_features=mf)
            gbr.fit(Xtrain_, ytrain_)
            ypred_gbr = gbr.predict(Xtest_)
            accu_gbr = r2_score(y_true=ytest_, y_pred=ypred_gbr)
            print("Gradient Boosting Forest accuracy: {:.2f}".format(accu_gbr))
            gbr_.append(accu_gbr)

            counter += 1

            print("\n", "-------------------------------------------------------------")

"""
Saving results
"""
grid_search_data = np.vstack([np.array(param_).T, np.array(rfr_), np.array(qfr_), np.array(gbr_)]).T
with open('grid_search_data.npy', 'wb') as fh:
    np.save(fh, grid_search_data)

"""
for Plotting
"""
start = time.time()
yestimated_qfr = qfr.predict(df_prediction, [0.05, 0.25, 0.50, 0.75, 0.95])
df_test['Recoveriness'] = yestimated_qfr[:, 2]
df_test['piw'] = yestimated_qfr[:, 4] - yestimated_qfr[:, 0]
stop = time.time()
print("\nPrediction: {:.6}s".format(stop - start))

df_test.to_csv(path_or_buf="df_test.csv", sep=",")

print("Done!")