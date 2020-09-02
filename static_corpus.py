# -*- coding: utf-8 -*-
# @Time : 2020/8/5 下午7:16
# @Author : chezhonghao
# @description : 1)分析语料
# @File : static_corpus.py
# @Software: PyCharm
import pandas as pd
from sklearn.model_selection import train_test_split
question_length=10
train_data_path='/home/chezhonghao/Projects/compatition/AutoMaster_TrainSet.csv'
def statics_corpus():
    train_df=pd.read_csv(train_data_path)
    trains,tests=train_test_split(train_df,test_size=0.1)
    print(trains.shape)
    print(tests.shape)
    brand=train_df['Brand']
    model=train_df['Model']
    question=train_df['Question']
    dialogue=train_df['Dialogue']
    report=train_df['Report']
    question_len_df=pd.DataFrame({'len':list(map(lambda x: len(str(x)),report))})
    k=question_len_df['len'].quantile(q=0.979)
    print(k)
def filter_data(data_dir='./data/vailed_data/vaild_data.csv'):
    data_all=pd.read_csv(data_dir)
    print(data_all.shape)
    data_all.dropna(how='any',inplace=True)
    data_all=data_all.reset_index(drop=True)
    print(data_all.shape)
    data_all.to_csv(data_dir,index=False)
# statics_corpus()
filter_data()