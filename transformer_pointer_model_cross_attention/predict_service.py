# -*- coding: utf-8 -*-
# @Time : 2020/8/31 上午8:30
# @Author : chezhonghao
# @description : 
# @File : predict_service.py
# @Software: PyCharm
import os
# os.environ['CUDA_VISIBLE_DEVICES']="-1"
from base_transformer_model.config import Config
from transformer_pointer_model_cross_attention.tpr_model import Transformer
import tensorflow as tf
from preprocess import load_vocab,encode_sentence
from transformer_pointer_model_cross_attention.utils import create_masks,create_look_ahead_mask,create_padding_mask

model=Transformer(num_layers=Config.num_layers, d_model=Config.d_model, num_heads=Config.num_heads, dff=Config.dff,
                  vocab_size=Config.vocab_size, batch_size=Config.batch_size,
                  rate=0.1)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=100000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
t2i,i2t=load_vocab(Config.vocab_path)
model.load_weights('../train_transformer_saved_weights/')
def create_mask(dialogue, question, report):
    # 编码器填充遮挡
    dialogue_padding_mask = create_padding_mask(dialogue)
    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    question_padding_mask = create_padding_mask(question)
    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(report)[1])
    dec_target_padding_mask = create_padding_mask(report)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return dialogue_padding_mask, combined_mask, question_padding_mask

def predict(inp_sentence,question):
    dialogue = encode_sentence(mode='dialogue',sentence=inp_sentence,token2id=t2i,maxlen=Config.max_len['dialogue'])
    question = encode_sentence(mode='question', sentence=question, token2id=t2i, maxlen=Config.max_len['question'])
    report=[t2i['<bos>']]
    # print(''.join([i2t[ids] for ids in report]))
    # start_token=[token for token in question]+[t2i['<bos>']]
    # print(''.join([i2t[ids] for ids in start_token]))
    dialogue_encoder = tf.expand_dims(dialogue, 0)
    question_decoder=tf.expand_dims(question,0)
    report_decoder=tf.expand_dims(report,0)
    # decoder_input = start_token
    # output = tf.expand_dims(decoder_input, 0)
    for i in range(Config.max_len['report']):
        dialogue_padding_mask, combined_mask, question_padding_mask = create_masks(dialogue_encoder, question_decoder,
                                                                                   report_decoder)

        predictions, attn_weights = model(dialogue_encoder, question_decoder,
                                          report_decoder, False,
                                          dialogue_padding_mask,
                                          combined_mask,
                                          dialogue_padding_mask,
                                          question_padding_mask)
        # print('report_decoder',''.join([i2t[idx] for idx in report_decoder[0].numpy()]))
        prediction_idx=predictions[0]
        # print('prediction_idx',''.join([i2t[idx] for idx in tf.argmax(prediction_idx,axis=-1).numpy()]))
        #找到最后一个要预测的id
        predicted_ids = prediction_idx[-1, :]  # (batch_size, 1, vocab_size)
        predicted_id=tf.argmax(predicted_ids)
        if predicted_id==t2i['<eos>']:
            return tf.squeeze(report_decoder, axis=0)
        predicted_id=tf.cast(tf.expand_dims([predicted_id],0),dtype=tf.int32)
        report_decoder=tf.concat([report_decoder, predicted_id], axis=-1)
    return tf.squeeze(report_decoder, axis=0)
if __name__ == '__main__':
    test_path='/home/chezhonghao/Projects/compatition/AutoMaster_TestSet.csv'
    import pandas as pd
    all_df=pd.read_csv(test_path)
    dialogue=list(all_df['Dialogue'])
    question=list(all_df['Question'])
    q_id=list(all_df['QID'])
    result={'QID':[],'Prediction':[]}
    for index in range(all_df.shape[0]):
        print(index)
        ids = predict(dialogue[index], question[index]).numpy()
        k = [i2t[i] for i in ids]
        prediction_model=''.join(k[1:])
        result['QID'].append(q_id[index])
        result['Prediction'].append(prediction_model)
    result_df=pd.DataFrame(result)
    result_df.to_csv('/home/chezhonghao/Projects/compatition/resluts.csv')
    # dialogue='技师说：你好，在大灯上面有两个孔，用螺丝刀插进去旋转可以调整'
    # question='大灯高度怎么 调节'
    # report='在大灯上面有两个孔，插进螺丝刀可以调整。'

    # print(ids)
    # k=[i2t[i] for i in ids]
    # print(''.join(k[1:]))