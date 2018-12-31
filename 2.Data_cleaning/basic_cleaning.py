


from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))


# checking missing data
def missing_table(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_df_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_df_data



# col_name='TARGET'
def FS_corr(df, col_name,num):
    df.corr()[col_name]
    #correlation matrix
    corrmat = df.corr()
    k = 10 #number of variables for heatmap
    cols = corrmat.nlargest(k, col_name)[col_name].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': num}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return cols


def pipe_cleandata(df):

    from sklearn.preprocessing import Imputer

    my_imp = Imputer()
    imp_X_train = my_imp.fit_transform(X_train)
    imp_X_test = my_imp.transform(X_test)
    print("Mean Absolute Error from Imputation:")
    print(score_dataset(imp_X_train, imp_X_test, y_train, y_test))

def missing_cat(df,target):
    miss_df=missing_table(df)
    lst=abs(corrmat[target])
    cat=[]
    for i in range(miss_df.shape[0]):
        if miss_df['Percent']>0.5 & lst[i]>0.05:
            cat[i]=1
        elif miss_df['Percent']>0.3 & lst[i]>=0.05:
            cat[i]=2
        elif miss_df['Percent']>0.3 & lst[i]<0.05:
            cat[i]=3
        else:
            cat[i]=4
    df=df.append(cat)
