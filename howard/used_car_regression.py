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
    생성할때, 전처리가 잘된(아웃라이어가 제거된) 데이터를 넣으시면 됩니다.
    """
    def __init__(self, df):
        self.df_origin = df
        self.df = df
        test = {}
        test2 = []
        end_num = 10
        start_num = 2
        
        for column in ['cylinders','manufacturer','title_status','type']:
            len_under_10 = len(self.df[column].value_counts()[(self.df[column].value_counts() < end_num) & (self.df[column].value_counts() > start_num)])
            if len_under_10:
                for i in range(len_under_10):
                    index = self.df[self.df[column] == self.df[column].value_counts()[(self.df[column].value_counts() < end_num) & (df[column].value_counts() > start_num)].index[i]].index.values
                    value = self.df[column].value_counts()[(self.df[column].value_counts() < end_num) & (df[column].value_counts() > start_num)].index[i]  
                    test[value] = index
        test2.append(test)
        
        index_df = pd.DataFrame(test2)
        
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
        포뮬러 식을 str타입으로 넣으시면 됩니다.
        
        return result, train_data, test_data
        result 객체와, train_data, test_data를 반환합니다
        """
#         X = self.df[self.df.columns.difference(['price'])]
#         Y = self.df['price']


        self.train_data, self.test_data = train_test_split(self.df, test_size = .20,random_state=random_state)
        
        self.train_data = pd.concat([self.train_data, self.df_origin[self.df_origin.index.isin([element for array in self.for_train_data_train for element in array])]])
        
        self.test_data = pd.concat([self.test_data, self.df_origin[self.df_origin.index.isin([element for array in self.for_train_data_test for element in array])]])
        
        
        model = sm.OLS.from_formula(formula, self.train_data)
        result = model.fit()
        return result, self.train_data, self.test_data

    
    def cross_validation(self,formula,random_state=0,cv=10):
        """
        "formula" => sm.OLS.from_formula("formula")
        포뮬러 식을 str타입으로 넣으시면 됩니다.
        교차검증된 값을 반환합니다.
        """
        kf = KFold(cv, shuffle=True, random_state=random_state)
        print(cv)
        model_cross_val_score = []
        for X_train_index, X_test_index in kf.split(self.train_data):

            X_train= self.train_data.iloc[X_train_index]
            X_test = self.train_data.iloc[X_test_index]

            X_train = pd.concat([X_train, self.df_origin[self.df_origin.index.isin([element for array in self.for_train_data_train for element in array])]], axis=0)
            
            X_test = pd.concat([X_test, self.df_origin[self.df_origin.index.isin([element for array in self.for_train_data_test for element in array])]], axis=0)
            
            model = sm.OLS.from_formula(formula, X_train)
            result = model.fit()           
            pred = result.predict(X_test)
            model_cross_val_score.append(r2_score(np.log(X_test.price),pred))
        return model_cross_val_score
