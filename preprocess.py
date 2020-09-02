# -*- coding: utf-8 -*-
# @Time : 2020/8/17 下午8:04
# @Author : chezhonghao
# @description : 1）制作词典；2）序列化tfrecord文件；3）解析tfrecord文件
# @File : preprocess.py
# @Software: PyCharm
import pandas as pd
import re
from collections import Counter
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

serilized_path = 'data/train_data/tfrecord/'
all_data_path = '/home/chezhonghao/Projects/compatition/AutoMaster_TrainSet.csv'
train_data_path='data/train_data/train_data.csv'
vaild_data_path='data/vailed_data/vaild_data.csv'
extern_vocab=['<pad>','<unk>','<sep>','<eos>','<bos>']
vocab_path='transformer_model/vocab.txt'
max_len={'question':152,'dialogue':779,'report':106,'brand':8,'base_transformer_model':19,'tar_report_input':106,'tar_report_output':106}
# max_len_vector_true={'question':152,'dialogue':779+1,'report':106+1,'brand':8+1,'base_transformer_model':19}
def split_dataset(all_dataset_path,train_data_path,vailed_data_path):
    '''
    切分数据集
    :param all_dataset_path: 原始数据集
    :param train_data_path: 训练集地址
    :param vailed_data_path: 验证集地址
    :return:
    '''
    all_df=pd.read_csv(all_dataset_path)
    train_df,vailed_df=train_test_split(all_df,test_size=0.05)
    train_df.to_csv(train_data_path,index=False)
    vailed_df.to_csv(vailed_data_path,index=False)
def create_vocab(data_path:str,extern_vocab:list,vocab_path:str):
    '''
    制作字典 charlevel
    :param data_path:原始训练数据地址
    :param extern_vocab:需要额外加入词典的词
    :return:
    '''
    def filter_dialogue(x):
        return str(re.sub(r'\s+','',str(x)))
    counter=Counter()
    train_df=pd.read_csv(data_path)
    brand=list(train_df['Brand'].apply(lambda x:re.sub(r'\s+','',str(x))))
    model=list(train_df['Model'].apply(lambda x:re.sub(r'\s+','',str(x))))
    question=list(train_df['Question'].apply(lambda x:str(x)))
    dialogue=list(train_df['Dialogue'].apply(filter_dialogue))
    report=list(train_df['Report'].apply(filter_dialogue))
    for index in range(len(question)):
        counter.update(question[index])
        counter.update(brand[index])
        counter.update(model[index])
        counter.update(dialogue[index])
        counter.update(report[index])
    print(len(counter))
    counter=dict(filter(lambda d: False if d[1] == 1 else True, counter.items()))
    sort_counter_list=sorted(counter.items(),key=lambda k:k[1],reverse=True)
    print(len(sort_counter_list))
    with open(vocab_path,'w',encoding='utf8') as f:
        for extern_token in extern_vocab:
            f.write(extern_token+'\n')
        for char_tuple in sort_counter_list:
            f.write(char_tuple[0]+'\n')
def load_vocab(vocab_path):
    '''
    加载vocab
    :param vocab_path:
    :return: id2token token2id
    '''
    tokens=[token for token in open(vocab_path,'r',encoding='utf8').read().splitlines()]
    id2token={id:token for id,token in enumerate(tokens)}
    token2id={token:id for id,token in enumerate(tokens)}
    return token2id,id2token
def encode_sentence(mode,sentence,token2id,maxlen):
    '''
    对”一句话“进行encode成词典中的id,根据maxlen对sentence进行剪裁
    :param mode: brand  base_transformer_model  question  dialogue  report对应于x的五个组成部分
    :param sentence:需要encode的sentence
    :param token2id:词典
    :param maxlen:每句话最大长度,这个值是和每个mode对应的，比如brand和question两者的maxlen肯定不一样
    :return:二维矩阵[[id_1,id_2,...,id_maxlen]]
    '''
    def cut_sentence(sequence):
        return tf.keras.preprocessing.sequence.pad_sequences(sequence,maxlen=maxlen,dtype='int32',
                                                      padding='post',truncating='post',value=0)
    if mode=='brand':
        sentence=['<bos>']+[token for token in sentence]+['<eos>']
        sentence_encode=[[token2id.get(char,token2id.get('<unk>')) for char in sentence]]
        sentence_encode=cut_sentence(sentence_encode)
        return np.squeeze(sentence_encode)
    if mode=='base_transformer_model':
        sentence=['<bos>']+[token for token in sentence]+['<eos>']
        sentence_encode=[[token2id.get(char,token2id.get('<unk>')) for char in sentence]]
        sentence_encode = cut_sentence(sentence_encode)
        return np.squeeze(sentence_encode)
    if mode == 'question':
        sentence = ['<bos>']+[token for token in sentence]+['<eos>']
        sentence_encode = [[token2id.get(char, token2id.get('<unk>')) for char in sentence]]
        sentence_encode = cut_sentence(sentence_encode)
        return np.squeeze(sentence_encode)
    if mode=='dialogue':
        sentence = ['<bos>']+[token for token in sentence]+['<eos>']
        sentence_encode = [[token2id.get(char, token2id.get('<unk>')) for char in sentence]]
        sentence_encode = cut_sentence(sentence_encode)
        return np.squeeze(sentence_encode)
    if mode=='tar_report_input':
        sentence = ['<bos>']+[token for token in sentence]
        sentence_encode = [[token2id.get(char, token2id.get('<unk>')) for char in sentence]]
        sentence_encode = cut_sentence(sentence_encode)
        return np.squeeze(sentence_encode)
    if mode == 'tar_report_output':
        sentence = [token for token in sentence]+['<eos>']
        sentence_encode = [[token2id.get(char, token2id.get('<unk>')) for char in sentence]]
        sentence_encode = cut_sentence(sentence_encode)
        return np.squeeze(sentence_encode)
    raise ('mode error, mode must be: brand  base_transformer_model question dialogue report')

def serilized_tfrecord(data_path,token2id,maxlen,serilized_path,n_shards,data_mod):
    '''
    将训练数据转成tfrecord并序列化,进行存储
    :param data_path: 要序列化的数据地址
    :param token2id:
    :param maxlen: 字典 记录每个特征最长长度 比如{‘brand’：8}
    :param serilized_path: 序列化数据存储地址,是目录不是文件
    :param data_mod:train or vailed
    :param n_shards:一共要存成几个文件
    :return:
    '''
    def filter_dialogue(dialogue):
        return str(dialogue)
    def create_feature(index,data_columns_list,data_list):
        '''
        生成feature 方便生成example使用
        :param index: 该条example的索引
        :param data_columns_list: 比如['brand,base_transformer_model','question','dialogue','report']
        :param data_list: 比如[brand,base_transformer_model,question,dialogue,report]
        :return:feature
        '''
        feature=dict()
        for columns_index in range(len(data_columns_list)):
             feature.update({data_columns_list[columns_index]:tf.train.Feature(
                 int64_list=tf.train.Int64List(value=encode_sentence(mode=data_columns_list[columns_index],
                                                                     sentence=data_list[columns_index][index],
                                                                     token2id=token2id,
                                                                     maxlen=maxlen[data_columns_list[columns_index]])))})
        return feature
    data_df = pd.read_csv(data_path)
    example_num=data_df.shape[0]
    steps_per_shard =example_num // n_shards
    brand = data_df['Brand'].apply(lambda x: re.sub(r'\s+', '', str(x)))
    model = list(data_df['Model'].apply(lambda x: re.sub(r'\s+', '', str(x))))
    question = list(data_df['Question'].apply(lambda x: str(x)))
    dialogue = list(data_df['Dialogue'].apply(filter_dialogue))
    report = list(data_df['Report'].apply(filter_dialogue))
    tar_report_input=report
    tar_report_output=report
    data_columns_list = ['brand','base_transformer_model','question','dialogue','tar_report_input','tar_report_output']
    data_list = [brand, model, question, dialogue,tar_report_input,tar_report_output]
    if not os.path.exists(serilized_path):
        os.mkdir(serilized_path)
    for shard_id in range(n_shards):
        base_name = '{}_{}_{}.tfrecords'.format(data_mod, n_shards, shard_id)
        full_path_name=os.path.join(serilized_path,base_name)
        with tf.io.TFRecordWriter(full_path_name) as writer:
            for i in range(shard_id*steps_per_shard,min((shard_id+1)*steps_per_shard,example_num)):
                if i % 1000==0:
                    logging.info('serilized example : %s' % (str(i)))
                features=tf.train.Features(
                    feature=create_feature(index=i,data_columns_list=data_columns_list,
                                           data_list=data_list)
                )
                example=tf.train.Example(features=features)
                writer.write(example.SerializeToString())
def tfrecord_reader(batch_size,serilized_path,max_len,n_readers=3,buffer_size=1000,num_parallel=3):
    '''
    读取并解析tfrecord文件
    :param serilized_path: trrecord存放文件目录
    :param max_len: 字典 存放每个feature所对应的最大长度 比如：{‘brand’:34}
    :param n_readers:并行读tfrecord文件的数量
    :return:
    '''
    serilized_path=serilized_path+'*.tfrecords'
    expect_example={
        'brand':tf.io.FixedLenFeature([max_len['brand']],dtype=tf.int64,
                                                   default_value=[0]*(max_len['brand'])),
        'base_transformer_model':tf.io.FixedLenFeature([max_len['base_transformer_model']],dtype=tf.int64,
                                                   default_value=[0]*(max_len['base_transformer_model'])),
        'question':tf.io.FixedLenFeature([max_len['question']],dtype=tf.int64,
                                                   default_value=[0]*(max_len['question'])),
        'dialogue':tf.io.FixedLenFeature([max_len['dialogue']],dtype=tf.int64,
                                                   default_value=[0]*(max_len['dialogue'])),
        'tar_report_input':tf.io.FixedLenFeature([max_len['report']],dtype=tf.int64,
                                                   default_value=[0]*(max_len['report'])),
        'tar_report_output': tf.io.FixedLenFeature([max_len['report']], dtype=tf.int64,
                                                  default_value=[10] * (max_len['report']))
    }
    def parse_tfrecord_line(serilized_example):
        '''
        解析一条example
        :param one_example:一条example
        :return:
        '''
        example=tf.io.parse_single_example(
            serialized=serilized_example,
            features=expect_example
        )
        for name in list(example.keys()):
            temp=example[name]
            if temp.dtype==tf.int64:
                temp=tf.cast(temp,tf.int32)
                example[name]=temp
        return example
    dataset=tf.data.Dataset.list_files(serilized_path)
    dataset.repeat()
    #注意看 这种模式
    dataset=dataset.interleave(
        lambda filename:tf.data.TFRecordDataset(filename),
        cycle_length=n_readers
    )
    dataset.shuffle(buffer_size=buffer_size)
    dataset=dataset.map(
        parse_tfrecord_line,
        num_parallel_calls=num_parallel
    )
    dataset=dataset.batch(batch_size=batch_size)
    dataset.repeat()
    return dataset
if __name__ == '__main__':
    #切分数据集
    # split_dataset(all_data_path,train_data_path,vaild_data_path)
    #建立词典
    # create_vocab(data_path=train_data_path,extern_vocab=extern_vocab,vocab_path=vocab_path)
    #encode_sentence
    t2i,i2t=load_vocab(vocab_path)
    # print(t2i['定'])
    # print(i2t[130])
    # kk=encode_sentence(mode='question',sentence='方向机重，助力泵，方向机都换了还是一样',token2id=t2i,maxlen=10)
    #序列化tfrecord文件
    # serilized_tfrecord(train_data_path, t2i, max_len,serilized_path,10,'train')
    # serilized_tfrecord('data/test_test/test_test_data.csv', t2i, max_len, 'data/test_test/tfrecord', 1, 'train')
    # serilized_tfrecord(vaild_data_path, t2i, max_len, 'data/vailed_data/tfrecord/', 1, 'vailed')
    #读取tfrecord文件并解析
    # data_batch=tfrecord_reader(12,serilized_path,max_len,n_readers=3,buffer_size=1000,num_parallel=3)
    # for k in data_batch:
    #     brand=k['tar_report_input'].numpy()
    #     print(''.join([i2t[idx] for idx in brand[0]]))
    # print(len(i2t))