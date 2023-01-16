from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

from category_encoders.target_encoder import TargetEncoder

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def load_dataset(train = True):
    
    if(train == True):
        values = pd.read_csv('./dataset/train_values.csv') 
        labels = pd.read_csv('./dataset/train_labels.csv')
        dataset = pd.merge(values, labels, how='inner', on = 'building_id')
    else:
        dataset = pd.read_csv('./dataset/test_values.csv')
        
    return dataset

def scale_features(features, method):
    scaled = None

    if(type(features) == type(pd.Series)):
        features = np.array(features).reshape(-1, 1)

    scaled = method.fit_transform(features)
   
    return scaled, method

def feature_compress(df):
    n_superstructure = np.array(df.loc[:, 'has_superstructure_adobe_mud':'has_superstructure_other']).sum(axis=-1)
    df['number_of_different_superstructures'] = n_superstructure

    n_secondary_uses = np.array(df.loc[:, 'has_secondary_use_agriculture':'has_secondary_use_other']).sum(axis=-1)
    df['number_of_secondary_uses'] = n_secondary_uses

    return df

def majority_vote(y_pred_list, weights):
    for i, weight in enumerate(weights):
        y_pred_list[i] = weight*y_pred_list[i]

    y_pred = np.argmax(y_pred_list.sum(axis = 0), axis = 1)+1

    return y_pred


def get_confusion_matrix(y_test, y_pred):
    plt.rcParams.update({'font.size': 22})
    fig,axs=plt.subplots(figsize=(10, 10))
    matrix = confusion_matrix(y_test, y_pred)

    df_cm = pd.DataFrame(matrix, columns=np.unique(y_pred), index = np.unique(y_pred))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm/np.sum(matrix), annot=True, 
                fmt='.2%', cmap='YlOrBr', ax=axs, annot_kws={"size": 16})

    return



class Preprocessor:
    def __init__(self, compress_cols = True, scaled_cols = None, target_cols = None, one_hot_cols = None, make_cat_cols = None):
    
        self.compress_cols_ = compress_cols
        self.scaled_cols_ = scaled_cols
        self.target_cols_ = target_cols
        self.one_hot_cols = one_hot_cols
        self.make_cat_cols = make_cat_cols
        self.scaler_ = None
        self.one_hot_encoder_ = None
        self.target_encoder_ = None

    def fit(self, X, y):
        if self.scaled_cols_ != None:
            self.scaler_ = RobustScaler()
            self.scaler_.fit(X[self.scaled_cols_])

        if self.one_hot_cols != None:
            self.one_hot_encoder_ = make_column_transformer((OneHotEncoder(), self.one_hot_cols), remainder='passthrough', verbose_feature_names_out=False)
            self.one_hot_encoder_.fit(X)

        if self.target_cols_ != None:
            self.target_encoder_ = TargetEncoder(cols=self.target_cols_, min_samples_leaf=20, smoothing=10)
            self.target_encoder_.fit(X, y)

        return

    def transform(self, X):


        if self.scaled_cols_ != None:
            scaled = self.scaler_.transform(X[self.scaled_cols_])
            X[self.scaled_cols_] = scaled

        if self.target_cols_ != None:
            X = self.target_encoder_.transform(X)


        if self.one_hot_cols != None:
            encoded = self.one_hot_encoder_.transform(X)
            X = pd.DataFrame(encoded, columns=self.one_hot_encoder_.get_feature_names_out())

        if self.compress_cols_:
            X=feature_compress(X)

        if self.make_cat_cols != None:
            for f in self.make_cat_cols:
                X[f] = pd.Series(X[f], dtype = 'category')

        return X