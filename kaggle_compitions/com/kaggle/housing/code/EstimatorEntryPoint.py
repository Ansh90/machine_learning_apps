# import lightgbm as lgb
# import xgboost as xgb
import os

import numpy as np
import pandas as pd
from StackingAveragedModel import StackingAveragedModel
from scipy.stats import skew
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from subprocess import check_output

class CSVReader:
    def getTrainingData(self):
        print(os.getcwd())
        pd.set_option('display.float_format',
                      lambda x: '{:.3f}'.format(x))  # Limiting floats output to 3 decimal points
        print(check_output(["ls", "./../../../../resources"]).decode("utf8"))  # check the files available in the directory
        train = pd.read_csv('./../../../../resources/train.csv', engine='python')
        # print("train : " + str(train.shape))
        return train

    def getTestData(self):
        # print(os.getcwd())
        pd.set_option('display.float_format',
                      lambda x: '{:.3f}'.format(x))  # Limiting floats output to 3 decimal points
        # print(check_output(["ls", "./../../../../resources"]).decode("utf8"))  # check the files available in the directory
        test = pd.read_csv('./../../../../resources/test.csv', engine='python')
        # print("test : " + str(test.shape))
        return test


class SimpleModelAssembler(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(model) for model in self.models]
        # for cloned_model in self.models:
        #     self.models_.append(clone(cloned_model))

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, testDataSet):
        predictions = np.column_stack([model.train_model(testDataSet) for model in self.models_])
        return np.mean(predictions, axis=1)


class DataPreprocessing():

    def process(self, all_data):
        self.all_data = all_data
        self.numericTranformation(self)
        self.catagoricalTransformation(self)

    def numericTranformation(self):
        # log transform skewed numeric features:
        numeric_feats = self.all_data.dtypes[self.all_data.dtypes != "object"].index

        skewed_feats = self.all_data[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
        skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
        skewed_feats = skewed_feats.index
        self.all_data[skewed_feats] = np.log1p(self.all_data[skewed_feats])

    def catagoricalTransformation(self):
        self.all_data = pd.get_dummies(self.all_data)
        # filling NA's with the mean of the column:
        self.all_data = self.all_data.fillna(self.all_data.mean())


class ModelAssemblerEntryPoint:
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=5)
    n_folds = 5

    def multiLayerAssembelModels(self):
        self.dataProcessing(self)
        # stacking model selection
        base_models = (self.ENet, self.GBoost, self.KRR, self.lasso)
        #models = (self.ENet, self.GBoost, self.KRR, self.lasso)
        stacked_averaged_models = StackingAveragedModel(base_models,meta_model=self.lasso)
        score = self.rmsle_cv(self, stacked_averaged_models, self.X_train, self.train_y)
        print(score)
        print("\n Mean Score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    def assembelModels(self):
        self.dataProcessing(self)
        # stacking model selection
        models = (self.ENet, self.GBoost, self.KRR, self.lasso)
        modelsAssembler = SimpleModelAssembler(models)
        score = self.rmsle_cv(self, modelsAssembler, self.X_train, self.train_y)
        print(score)
        print("\n Mean Score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



    # rmsle_cv has cross_val_score method which internally calls fit and predict method of model to train
    # and make prediction
    # root mean square logrithmic error
    def rmsle_cv(self, model, X_train, y_train):
        #todo: change
        #kf = KFold(self.n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
        kf = KFold(self.n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf))
        return (rmse)

    def rmsle(y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))

    def dataProcessing(self):
        csvObj = CSVReader
        train = csvObj.getTrainingData(csvObj)
        test = csvObj.getTestData(csvObj)

        # Concat all the columns except index and target in all_data
        all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']))
        # train["SalePrice"] = np.log1p(train["SalePrice"])

        dataProcess = DataPreprocessing
        dataProcess.process(dataProcess, all_data)

        # creating matrices for sklearn:
        self.X_train = dataProcess.all_data[:train.shape[0]].values
        self.train_y = np.log1p(train.SalePrice).values

class EstimatorEntryPoint:

    def simpleModelKickOff():
        print("AI started thinking")
        modelObj = ModelAssemblerEntryPoint
        modelObj.assembelModels(modelObj)

    def advanceModelKickOff():
        print("AI started thinking")
        modelObj = ModelAssemblerEntryPoint
        modelObj.multiLayerAssembelModels(modelObj)


    #simpleModelKickOff()
    advanceModelKickOff()
    print("Its time to take some rest AI :) ")
