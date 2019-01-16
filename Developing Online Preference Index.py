import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from my_utils import *
from collections import defaultdict
import matplotlib.font_manager as fm


# DIR 설정
data_dir = 'C:\\Users\\CSH\\Desktop\\lpoint\\data\\'

# 선호지수 개발 시작 -  검색어와 구매 브랜드 유사성 (선호브랜드가 뚜렷한가의 정도)
product = pd.read_csv(data_dir + 'product.csv')
search_1 = pd.read_csv(data_dir + 'Search1.csv')
master = pd.read_csv(data_dir + 'Master.csv')

data = product.merge(search_1, on=['CLNT_ID', 'SESS_ID'], how='left').merge(master, on='PD_C', how='left')
data = data[['CLNT_ID', 'SESS_ID', 'PD_BRA_NM', 'PD_BUY_AM', 'PD_BUY_CT', 'KWD_NM', 'SEARCH_CNT', 'CLAC1_NM', 'CLAC2_NM', 'CLAC3_NM']]

data = data[~data['KWD_NM'].isnull()]
data['PD_BRA_NM_LIST'] = data['PD_BRA_NM'].apply(lambda x : x.replace("("," ").replace(")"," ").replace("]"," ").replace("["," ").split())
data['KWD_NM'] = data['KWD_NM'].apply(lambda x : x.split())

def calculating_pref_index(data):
    data = data.copy()
    data_np = np.array(data[['PD_BRA_NM_LIST', 'KWD_NM']])
    scores = []
    for i in range(len(data_np)):
        row = data_np[i]
        ans = 0
        for brands_word in row[0]:
            for keywords_word in row[1]:
                if keywords_word in brands_word:
                    ans += 1
                if brands_word in keywords_word:
                    ans += 1
        scores.append(ans)
    data['SCORE'] = scores
    pref_index = data.groupby(['CLAC1_NM'])['SCORE'].sum() / data.groupby(['CLAC1_NM'])['SCORE'].count()
    return pref_index.sort_values(ascending=False)

preference_index = calculating_pref_index(data)
pd.DataFrame(preference_index).to_csv("Preference_Index.csv", index=False) # 결과물 저장