
import pandas as pd

class setup_data(object):
    def __init__(self, *, path):
        self.__path = path
        self.__application_train = None
        self.__application_test = None
        self.__sample_submission = None
        self.__application_train_feature = None
        self.__application_train_label = None
        self.__application_test_feature = None

        self.__categorical_columns = None
        self.__numeric_columns = None

        self.__categorical_index = None

        self.__clf = None


    path = "/Users/David/Desktop/0.Home\ default\ risk/all"
