import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from my_utils import *
from wordcloud import WordCloud
from collections import defaultdict, Counter
import matplotlib.font_manager as fm


# DIR 설정
data_dir = 'C:\\Users\\CSH\\Desktop\\lpoint\\data\\'

# 데이터들 읽어오기
product = pd.read_csv(data_dir + 'product.csv')
search_1 = pd.read_csv(data_dir + 'Search1.csv')
search_2 = pd.read_csv(data_dir + 'Search2.csv')
custom = pd.read_csv(data_dir + 'custom.csv')
session = pd.read_csv(data_dir + 'session.csv')
master = pd.read_csv(data_dir + 'master.csv')

# product 변수 정리
product.drop(['HITS_SEQ', 'PD_ADD_NM', 'PD_BRA_NM'], axis=1, inplace=True)
product['CLNT_ID'] = product['CLNT_ID'].apply(lambda x: str(x))
product['SESS_ID'] = product['SESS_ID'].apply(lambda x: str(x))
product['PD_BUY_AM'] = product['PD_BUY_AM'].apply(lambda x: int((x).replace(",","")))
product['PD_BUY_CT'] = product['PD_BUY_CT'].apply(lambda x: int(str(x).replace(",","")))

# search_1 변수 정리
search_1['CLNT_ID'] = search_1['CLNT_ID'].apply(lambda x: str(x))
search_1['SESS_ID'] = search_1['SESS_ID'].apply(lambda x: str(x))

# search_2 변수정리
search_2['SESS_DT'] = pd.to_datetime(search_2['SESS_DT'], format="%Y%m%d")
search_2['SEARCH_CNT'] = search_2['SEARCH_CNT'].apply(lambda x: int(x.replace(",","")))

# custom 변수 정리
custom['CLNT_ID'] = custom['CLNT_ID'].apply(lambda x : str(x))
custom['CLNT_AGE'] = custom['CLNT_AGE'].apply(lambda x : str(int(x)))

# session 변수 정리
session.drop('SESS_SEQ', axis=1, inplace=True)
session['CLNT_ID'] = session['CLNT_ID'].apply(lambda x: str(x))
session['SESS_ID'] = session['SESS_ID'].apply(lambda x: str(x))
session['SESS_DT'] = pd.to_datetime(session['SESS_DT'], format="%Y%m%d")
page_mean = session['TOT_PAG_VIEW_CT'].mean()
sess_mean = session['TOT_SESS_HR_V'].dropna().apply(lambda x : int(x.replace(",",""))).mean() # NA 자리에는 평균값 채워넣기
sess_mean = int(sess_mean)
session['TOT_PAG_VIEW_CT'][session['TOT_PAG_VIEW_CT'].isnull()] = page_mean
session['TOT_PAG_VIEW_CT'] = session['TOT_PAG_VIEW_CT'].apply(lambda x : int(x))
session['TOT_SESS_HR_V'][session['TOT_SESS_HR_V'].isnull()] = sess_mean
session['TOT_SESS_HR_V'] = session['TOT_SESS_HR_V'].apply(lambda x : int(str(x).replace(",","")))

# master 변수 정리
master.drop('PD_NM', axis=1, inplace=True)
master.drop(['CLAC2_NM', 'CLAC3_NM'], axis=1, inplace=True)

# 하나의 테이블로 합치기

# search_1 데이터를 [clnt id, sess id] 기준으로 묶기 ( 다른 데이터와 동일하게 바꿔줘야함 )
search_1['KWD_CNT'] = list(zip(search_1['KWD_NM'], search_1['SEARCH_CNT']))
search_1_sess = search_1.groupby(['CLNT_ID', 'SESS_ID'], as_index=False)['KWD_CNT'].agg(lambda x : sorted(x, key=lambda y : -y[1])[0])
search_1_sess['KWD_CNT'] = search_1_sess['KWD_CNT'].apply(lambda x: x[0]) # 가장 많이 언급된 검색어를 가져왔음

# 합치자
dat = product.merge(master, how='left', on='PD_C') # product + master
dat = dat[dat['CLAC1_NM'].isin(['남성의류', '스포츠패션', '여성의류', '패션잡화', '화장품/뷰티케어'])] # 주요품목만 살려놓기
dat = dummy_transform(dat, ['CLAC1_NM'])
dat.drop('PD_C', axis=1, inplace=True)
for col in dat.columns[-5:]:
    dat[col] = dat[col] * dat['PD_BUY_CT']
dat.drop('PD_BUY_CT', axis=1, inplace=True)
dat = dat.groupby(['CLNT_ID', 'SESS_ID'], as_index=False).sum() # product 세션 기준으로 변형 완성, 제품을 주요 품목 5개를 더미 * 구매액수로 변형했음
dat = dat.merge(session, on=['CLNT_ID', 'SESS_ID'], how='left')\
.merge(search_1_sess, on=['CLNT_ID', 'SESS_ID'], how='left') # + session, search_1_sess 맵핑

# Month 열 생성
dat['SESS_MONTH'] = dat['SESS_DT'].apply(lambda x: str(x)[:7])
dat.drop(['SESS_ID', 'SESS_DT'], axis=1, inplace=True)

# client, sess_month 기준으로 그룹바이 할 때 search_1 KWD_NM, ctg_nm, zon_nm, city_nm 은 가장 흔한것만 살려놓기
sum_col = dat.columns[1:9]
freq_col = dat.columns[9:13]
data_1 = dat.groupby(['CLNT_ID', 'SESS_MONTH'], as_index=False)[sum_col].sum()
data_2 = dat.groupby(['CLNT_ID', 'SESS_MONTH'], as_index=False)[freq_col].agg(
    lambda x : x.value_counts(dropna = True).index[0] if len(x) != x.isnull().sum() else None) # NA가 껴있으면 제외하고 count하기, NA만 있으면 NA 반환.
data = data_1.merge(data_2, on=['CLNT_ID', 'SESS_MONTH'], how='left')

# custom은 client, sess_month 기준으로 합치고 난뒤에 맵핑해주자
data = data.merge(custom, on='CLNT_ID', how='left')

# 원핫인코딩, 고빈도 팩터라이징
data_pre = top_factorize(data, 100, ['DVC_CTG_NM', 'ZON_NM', 'CITY_NM', 'KWD_CNT', 'CLNT_AGE'])
data_pre = dummy_transform(data_pre, ['CLNT_GENDER'])
data_pre.drop('CLNT_GENDER_M', axis=1, inplace=True)

# 휴가철 변수
data_pre['is_vacation'] = data_pre['SESS_MONTH'].apply(lambda x : 1 if x == '2018-08' else 0)

# 스케일링(표준화)
data_pre = standardize(data_pre, ['TOT_SESS_HR_V', 'TOT_PAG_VIEW_CT'])

# 월 변수 정수화
data_pre['SESS_MONTH'] = data_pre['SESS_MONTH'].apply(lambda x : int(x[-1]))

# 다음 월에 주요 품목을 구매 했는지 여부를 추가 (5개의 열 추가)
tmp = data_pre[['CLNT_ID', 'SESS_MONTH', 'CLAC1_NM_남성의류', 'CLAC1_NM_스포츠패션', 'CLAC1_NM_여성의류', 'CLAC1_NM_패션잡화', 'CLAC1_NM_화장품/뷰티케어']]
tmp['SESS_MONTH'] = tmp['SESS_MONTH'].apply(lambda x: x-1)
tmp.columns = ['CLNT_ID', 'SESS_MONTH', 'NEXT_MONTH_남성의류', 'NEXT_MONTH_스포츠패션', 'NEXT_MONTH_여성의류', 'NEXT_MONTH_패션잡화', 'NEXT_MONTH_화장품/뷰티케어']
data_pre = data_pre.merge(tmp, on=['CLNT_ID', 'SESS_MONTH'], how='left').fillna(0)
for col in  ['NEXT_MONTH_남성의류', 'NEXT_MONTH_스포츠패션', 'NEXT_MONTH_여성의류', 'NEXT_MONTH_패션잡화', 'NEXT_MONTH_화장품/뷰티케어']:
    data_pre[col] = data_pre[col].astype(int)
    data_pre[col] = data_pre[col].apply(lambda x : 1 if x != 0 else 0) # binary 로 !
data_pre = data_pre[data_pre['SESS_MONTH'] != 9] # 9월은 다음달 구매데이터가 없으니까 제거

# 이전월 기록 넣기
data_tmp = data_pre[['CLNT_ID', 'SESS_MONTH', 'PD_BUY_AM', 'CLAC1_NM_남성의류', 'CLAC1_NM_스포츠패션', 'CLAC1_NM_여성의류', 'CLAC1_NM_패션잡화', 'CLAC1_NM_화장품/뷰티케어']]
data_tmp['SESS_MONTH'] = data_tmp['SESS_MONTH'].apply(lambda x: x + 1)
data_tmp.columns = ['CLNT_ID', 'SESS_MONTH', 'PREV_MONTH_PD_BUY_AM', 'PREV_MONTH_남성의류', 'PREV_MONTH_스포츠패션', 'PREV_MONTH_여성의류', 'PREV_MONTH_패션잡화', 'PREV_MONTH_화장품/뷰티케어']
data_pre = data_pre.merge(data_tmp, on=['CLNT_ID', 'SESS_MONTH'], how='left')
data_pre = data_pre[data_pre['SESS_MONTH'] != 4].fillna(0) # 다 NA일것! 짜르자 !

data_pre.to_csv("data.csv", index=False) # 해당 디렉토리에 결과물 파일을 저장