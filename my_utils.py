import pandas as pd
import numpy as np
import pickle
import xgboost

class Balancing_Ensemble:

    def __init__(self, folder_name):
        self.folder_name = folder_name

    def fit(self, down_sample_ratio, up_sample_ratio, n_model, X, y):  # X는 np.matrix, y는 pd.Series 형태
        for i in range(n_model):
            xgb = xgboost.XGBClassifier(max_depth=4, nthread=4, colsample_bytree=0.8, colsample_bylevel=0.9,
                                      min_child_weight=10, n_jobs=4)
            y.reset_index(drop=True, inplace=True)
            one_idxes = y[y == 1].index
            one_idxes = np.random.choice(one_idxes, int(len(one_idxes) * up_sample_ratio))
            zero_idxes = y[y == 0].index
            zero_idxes = np.random.choice(zero_idxes, int(len(zero_idxes) * down_sample_ratio))
            train_x = X[np.concatenate([one_idxes, zero_idxes])]
            train_y = y[np.concatenate([one_idxes, zero_idxes])]
            xgb.fit(train_x, train_y)
            pickle.dump(xgb, open(self.folder_name + 'xgb_{}.pickle'.format(i), "wb"))

    def predict(self, X, n_model, threshold=2):
        record = np.zeros(n_model * X.shape[0]).reshape(n_model, X.shape[0])
        for i in range(n_model):
            model = pickle.load(open(self.folder_name + 'xgb_{}.pickle'.format(i), 'rb'))
            tmp = model.predict(X)
            record[i] = tmp.copy()
        record = record.sum(axis=0)
        record[record < n_model / threshold] = 0
        record[record >= n_model / threshold] = 1
        return record


def chk_dist(data, string):
    print("평균 %s는 %.3f 건, 상위 99%%의 %s는 %d 건이며 최고 %s는 %d 건이다."\
          %(string, data.mean(), string, data.quantile(0.99), string, data.max()))

def standardize(data, cols):
    data = data.copy()
    for col in cols:
        mean = data[col].mean()
        sd = data[col].var()**(1/2)
        data[col] = data[col].apply(lambda x : (x - mean)/sd)
    return data


def dummy_transform(data, columns):
    dummied= pd.get_dummies(data[columns])
    other_cols = data.loc[:, ~data.columns.isin(columns)]
    result = pd.concat([other_cols, dummied ], axis=1)
    return result


def top_factorize(data, n, cols): # 빈도가 top n 개인 변수까지는 Factorizing ! 그 외에 변수들은 NA 와 똑같이 -99로 처리 !
    data = data.copy()
    for col in cols:
        dic = dict(data[col].value_counts())
        dic_items = list(dic.items())
        dic_items.sort(key = lambda x : -x[1])
        tmp = data[col].copy()
        data[col] = None
        for item in dic_items[:n]:
            frequent_value = item[0]
            data[col][tmp == frequent_value] = tmp[tmp == frequent_value]
        data[col], _ = data[col].factorize(na_sentinel=-99)
    return data

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist