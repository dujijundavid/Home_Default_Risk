# coding:utf-8

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

class XgBoostBaseline(object):
    def __init__(self, *, path):
        # Path, train&test dataset, submission dataset
        self.__path = path
        self.__train = None
        self.__test = None
        self.__sample_submission = None

        # Features & Target variable
        self.__train_feature = None
        self.__train_label = None
        self.__test_feature = None

        # matrix for data storage
        self.__train_dmatrix = None
        self.__test_dmatrix = None

        # Parameter& classifer for Xgboost
        self.__params = {}
        self.__clf = None

    def data_prepare(self):
        self.__train = pd.read_csv(os.path.join(self.__path, "application_train.csv"))
        self.__test = pd.read_csv(os.path.join(self.__path, "application_test.csv"))
        self.__sample_submission = pd.read_csv(os.path.join(self.__path, "sample_submission.csv"))

        self.__train = self.__train.drop("SK_ID_CURR", axis=1)
        self.__test = self.__test.drop("SK_ID_CURR", axis=1)

        self.__train_feature = self.__train[[i for i in self.__train.columns if i != "TARGET"]]
        self.__train_label = self.__train["TARGET"]
        self.__test_feature = self.__test

        self.__train_feature = self.__train_feature.drop(
            self.__train_feature.select_dtypes(include=["object"]).columns.tolist(),
            axis=1
        )
        self.__test_feature = self.__test_feature.drop(
            self.__test_feature.select_dtypes(include=["object"]).columns.tolist(),
            axis=1
        )

        self.__train_dmatrix = xgb.DMatrix(
            data=self.__train_feature,
            label=self.__train_label,
            missing=np.nan,
            feature_names=self.__train_feature.columns
        )
        self.__test_dmatrix = xgb.DMatrix(
            data=self.__test_feature,
            missing=np.nan,
            feature_names=self.__test_feature.columns
        )

    def model_fit(self,num):
        self.__params["tree_method"] = "hist"
        self.__clf = xgb.train(self.__params, self.__train_dmatrix, num_boost_round=num)

    def model_predict(self):

        self.__sample_submission["TARGET"] = np.clip(self.__clf.predict(self.__test_dmatrix), 0, 1)
        self.__sample_submission.to_csv(
            '/Users/David/Desktop/0.Home default risk/submission/xgboost_baseline',
            index=False
        )

    def model_plot(self):
        xgb.plot_importance(self.__clf)
        plt.show()


if __name__ == "__main__":

    xbb = XgBoostBaseline(path="/Users/David/Desktop/0.Home default risk/all/")
    xbb.data_prepare()
    xbb.model_fit(100)
    xbb.model_predict()
    xbb.model_plot()