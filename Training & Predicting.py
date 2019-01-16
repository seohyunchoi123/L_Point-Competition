import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from my_utils import *
from wordcloud import WordCloud
from collections import defaultdict, Counter
import matplotlib.font_manager as fm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split


# DIR 설정
data_dir = 'C:\\Users\\CSH\\Desktop\\lpoint\\data\\'
data = pd.read_csv("data.csv", encoding='cp949')

# 트레인, 테스트셋 나누기
y_columns = ['NEXT_MONTH_남성의류', 'NEXT_MONTH_스포츠패션', 'NEXT_MONTH_여성의류', 'NEXT_MONTH_패션잡화', 'NEXT_MONTH_화장품/뷰티케어']
x_columns = np.array(list(set(data.columns) - set(y_columns)))
x_columns = x_columns[x_columns != 'CLNT_ID']
x_columns = sorted(x_columns)
index = data.index
train_idx = np.random.choice(index, int(len(data)*0.9), replace=False)
test_idx = list(set(index) - set(train_idx))
train = data.loc[train_idx,:]
test = data.loc[test_idx,:]

train_x_mat = np.array(train[x_columns])
test_x_mat = np.array(test[x_columns])

# 학습 시작 
model = Balancing_Ensemble(folder_name = './남성의류/')
model.fit(up_sample_ratio=0.5, down_sample_ratio= 0.05, n_model=100, X=train_x_mat, y=train[y_columns[0]])

model_1 = Balancing_Ensemble(folder_name = './스포츠패션/')
model_1.fit(up_sample_ratio=0.5, down_sample_ratio= 0.05, n_model=100, X=train_x_mat, y=train[y_columns[1]])

model_2 = Balancing_Ensemble(folder_name = './여성의류/')
model_2.fit(up_sample_ratio=0.5, down_sample_ratio= 0.05, n_model=100, X=train_x_mat, y=train[y_columns[2]])

model_3 = Balancing_Ensemble(folder_name = './패션잡화/')
model_3.fit(up_sample_ratio=0.5, down_sample_ratio= 0.05, n_model=100,  X=train_x_mat, y=train[y_columns[3]])

model_4 = Balancing_Ensemble(folder_name = './화장품뷰티케어/')
model_4.fit(up_sample_ratio=0.5, down_sample_ratio= 0.05, n_model=100, X=train_x_mat, y=train[y_columns[4]])

# 예측하여 결과를 프린트
threshold = 1
pred_0 = model.predict(X = test_x_mat, n_model = 100, threshold = threshold)
print("F1 Score : {}".format(f1_score(pred_0, test[y_columns[0]])))
print(pd.crosstab(pred_0, test[y_columns[0]]))

pred_1 = model_1.predict(X = test_x_mat, n_model = 100, threshold = threshold)
print("F1 Score : {}".format(f1_score(pred_1, test[y_columns[1]])))
print(pd.crosstab(pred_1, test[y_columns[1]]))

pred_2 =   model_2.predict(X = test_x_mat, n_model = 100, threshold = threshold)
print("F1 Score : {}".format(f1_score(pred_2, test[y_columns[2]])))
print(pd.crosstab(pred_2, test[y_columns[2]]))

pred_3 = model_3.predict(X = test_x_mat, n_model = 100, threshold = threshold)
print("F1 Score : {}".format(f1_score(pred_3, test[y_columns[3]])))
print(pd.crosstab(pred_3, test[y_columns[3]]))

pred_4 = model_4.predict(X = test_x_mat, n_model = 100, threshold = threshold)
print("F1 Score : {}".format(f1_score(pred_4, test[y_columns[4]])))
print(pd.crosstab(pred_4, test[y_columns[4]]))
