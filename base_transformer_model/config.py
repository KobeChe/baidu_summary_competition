# -*- coding: utf-8 -*-
# @Time : 2020/8/18 下午8:12
# @Author : chezhonghao
# @description : 
# @File : config.py
# @Software: PyCharm
class Config():
    d_model=512
    num_layers=6
    num_heads=8
    dff=1024
    vocab_size=3918
    pe_encoder_input=779
    pe_decoder_input=260
    epoches=1000
    serilized_path='../data/train_data/tfrecord/'
    serilized_path_eval='../data/vailed_data/tfrecord/'
    max_len = {'question': 152, 'dialogue': 779, 'report': 106, 'brand': 8, 'base_transformer_model': 19}
    batch_size=6
    eval_batch_size=32
    saved_model_path = '../saved_transformer_model'
    train_num=84648
    steps_per_epoch=5200
    vocab_path='../transformer_model/vocab.txt'
    log_name='./log.txt'
    ckpt='../mytransformer_checkpoints'
    train_ckpt='../train_mytransformer_checkpoints'