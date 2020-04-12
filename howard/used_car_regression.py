import pandas as pd
import statsmodels.api as sm
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


class UsedCarRegression:
    """
    initializer needs DataFrame which is filtered by data from 'vinaudit.com' api.
    객체를 생성할때, 전처리가 잘된(아웃라이어가 제거된) 데이터를 넣으시면 됩니다.
    """
    def __init__(self, df):
        self.df_origin = df
        self.df = df
        
        
        len_under_10_dict = {}
        len_under_10_list = []
        end_num = 10
        start_num = 2
        
        for column in ['cylinders','manufacturer','title_status','type']:
            len_under_10 = len(self.df[column].value_counts()[(self.df[column].value_counts() < end_num) & (self.df[column].value_counts() > start_num)])
            if len_under_10:
                for i in range(len_under_10):
                    index = self.df[self.df[column] == self.df[column].value_counts()[(self.df[column].value_counts() < end_num) & (df[column].value_counts() > start_num)].index[i]].index.values
                    value = self.df[column].value_counts()[(self.df[column].value_counts() < end_num) & (df[column].value_counts() > start_num)].index[i]  
                    len_under_10_dict[value] = index
        len_under_10_list.append(len_under_10_dict)
        
        index_df = pd.DataFrame(len_under_10_list)
        
        self.for_test_data = []
        self.for_train_data_train = []
        self.for_train_data_test = []
        for column in index_df.columns:
            start = list(index_df[column][0])
            random.shuffle(start)
            if len(start) > 4:
                m = [start[i:i + 3] for i in range(0, len(start), 3)]
                self.for_test_data.append(m[0])
                self.for_train_data_train.append(m[1])
                self.for_train_data_test.append(m[2])
            elif len(start) == 4:
                m = [start[:2], start[2:3], start[3:]]
                self.for_test_data.append(m[0])
                self.for_train_data_train.append(m[1])
                self.for_train_data_test.append(m[2])
            else :
                m = [[i] for i in start]
                self.for_test_data.append(m[0])
                self.for_train_data_train.append(m[1])
                self.for_train_data_test.append(m[2])
    
        # 10개 미만 삭제
        for column in self.df.columns.difference(['id','price','odometer','year']):
            values = [value for value in df[column].value_counts()[self.df[column].value_counts() < 10].keys()]
            if values:
                for value in values:
                    self.df = self.df[self.df[column] != value]

#         # 데이터 분할
#         self.train_data, self.test_data = train_test_split(self.df, test_size = .20, random_state = random_state)
#         self.train_data = pd.concat([self.train_data, self.df_origin.iloc[
#             [element for array in self.for_train_data_train for element in array] + [element for array in self.for_train_data_test for element in array]
#         ]],axis=0)
#         self.test_data = pd.concat([self.test_data, self.df_origin.iloc[
#             [element for array in self.for_test_data for element in array]]])

                
        
    def model_fit(self, formula, random_state=0):
        """
        "formula" => sm.OLS.from_formula("formula")
        return : result, train_data, test_data, 포뮬러 식을 str타입으로 넣으면, result 객체와, train_data, test_data를 반환합니다
        """
#         X = self.df[self.df.columns.difference(['price'])]
#         Y = self.df['price']
        self.train_ls = self.for_train_data_train
        self.test_ls = self.for_train_data_test
        

        self.train_data, self.test_data = train_test_split(self.df, test_size = .20,random_state=random_state)
        
        self.train_data = pd.concat([self.train_data, self.df_origin[self.df_origin.index.isin([element for array in self.for_train_data_train for element in array])]])
        
        self.test_data = pd.concat([self.test_data, self.df_origin[self.df_origin.index.isin([element for array in self.for_train_data_test for element in array])]])
        
        
        model = sm.OLS.from_formula(formula, self.train_data)
        result = model.fit()
        return result, self.train_data, self.test_data, self.train_ls, self.test_ls

    
    def cross_validation(self,formula,random_state=0,cv=10):
        """
        "formula" => sm.OLS.from_formula("formula"), 포뮬러 식을 str타입으로 넣으면 교차검증된 값을 list로 반환합니다.
        """
        kf = KFold(cv, shuffle=True, random_state=random_state)
        model_cross_val_score = []
        for X_train_index, X_test_index in kf.split(self.train_data):

            X_train= self.train_data.iloc[X_train_index]
            X_test = self.train_data.iloc[X_test_index]

            X_train = pd.concat([X_train, self.df_origin[self.df_origin.index.isin([element for array in self.for_train_data_train for element in array])]], axis=0)
            
            X_test = pd.concat([X_test, self.df_origin[self.df_origin.index.isin([element for array in self.for_train_data_test for element in array])]], axis=0)
            
            model = sm.OLS.from_formula(formula, X_train)
            result = model.fit()           
            pred = result.predict(X_test)           
            r2 = r2_score(np.log(X_test.price),pred)
            
            n = X_test.shape[0] 
            p = pd.get_dummies(X_test).shape[1]
            adj_r = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
            
            model_cross_val_score.append(adj_r)
        return model_cross_val_score

    
    
    def regularized_method(self,formula,random_state=0,cv=10, alpha=0.0001, L1_wt=0):
        """
        "formula" => sm.OLS.from_formula("formula"), 포뮬러 식을 str타입으로 넣으면 교차검증된 값을 list로 반환합니다.
        """
        kf = KFold(cv, shuffle=True, random_state=random_state)
        model_cross_val_score = []
        for X_train_index, X_test_index in kf.split(self.train_data):

            X_train= self.train_data.iloc[X_train_index]
            X_test = self.train_data.iloc[X_test_index]

            X_train = pd.concat([X_train, self.df_origin[self.df_origin.index.isin([element for array in self.for_train_data_train for element in array])]], axis=0)
            
            X_test = pd.concat([X_test, self.df_origin[self.df_origin.index.isin([element for array in self.for_train_data_test for element in array])]], axis=0)
            
            model = sm.OLS.from_formula(formula, X_train)
            result = model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
            pred = result.predict(X_test)           
            r2 = r2_score(np.log(X_test.price),pred)
            
            n = X_test.shape[0] 
            p = pd.get_dummies(X_test).shape[1]
            adj_r = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
            
            model_cross_val_score.append(adj_r)
        return np.mean(model_cross_val_score)
    
    
    