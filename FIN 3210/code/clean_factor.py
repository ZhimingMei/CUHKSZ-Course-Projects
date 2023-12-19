import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 中位数去极值法 并且标准化
def filter_MAD(df,factor,n=5):
    # 去极值
    median = df[factor].quantile(0.5)
    new_median =((df [factor]- median).abs()).quantile(0.5)
    max_range = median + n * new_median
    min_range = median - n * new_median
    for i in range(df.shape[0]):
        if df.loc[i, factor] > max_range:
            df.loc[i, factor] = max_range
        elif df.loc[i, factor] < min_range:
            df.loc[i,factor] = min_range

    #标准化
    df[factor] = (df[factor] - df[factor].mean()) / df[factor].std()

    return df


df = pd.read_table('D:/Desktop/厚方 投资建模/merged_daily_freq_factor.txt',sep=',')
df = df.sort_values(by = 'TRADE_DT')

grouped = df.groupby('TRADE_DT')

factor = 'S_VAL_PE,S_VAL_PB_NEW,S_VAL_PS,S_DQ_TURN,S_DQ_MV,NET_ASSETS_TODAY,NET_PROFIT_PARENT_COMP_TTM,NET_CASH_FLOWS_OPER_ACT_TTM,OPER_REV_TTM,Variance20,Skewness20,Kurtosis20,SharpeRatio20,VOL20,VSTD20,TVMA20,WVAD'
factor_list = factor.split(',')
L=[]

for i in grouped:
    a = i[1]
    a = a.reset_index(drop=True)
    a = a.fillna(a.mean()) # 以日度数据为单位，以均值填充为
    y = a['S_DQ_MV'] # 市值为应变量

    for factor in factor_list:
        a = filter_MAD(a, factor)

    x = a[factor].values.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(x, y)  # 拟合
    y_predict = lr.predict(x)
    df[factor] = y - y_predict
    L.append(a)
    # print(a)
    # break

# 数据合并  
b = L[0]
for i in L[1:]:
    b = b.append(i)
b = b.reset_index(drop=True)

print(b.info())
b.to_csv('merged_daily_freq_factor_cleaned.gz', compression='gzip', index=False)

