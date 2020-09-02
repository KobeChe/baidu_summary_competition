# -*- coding: utf-8 -*-
# @Time : 2020/8/21 下午2:34
# @Author : chezhonghao
# @description : 
# @File : predict_service.py
# @Software: PyCharm
import os
os.environ['CUDA_VISIBLE_DEVICES']="-1"
from base_transformer_model.config import Config
from preprocess import tfrecord_reader,encode_sentence,load_vocab
from base_transformer_model.component import create_look_ahead_mask,create_padding_mask
import tensorflow as tf
model=tf.saved_model.load('./saved_transformer_model',tags=[tf.saved_model.SERVING])
# inference=base_transformer_model.signatures['serving_default']
t2i, i2t = load_vocab('./transformer_model/vocab.txt')
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
def predict(inp_sentence,question,report):
    dialogue=encode_sentence(mode='dialogue',sentence=inp_sentence,token2id=t2i,maxlen=Config.max_len['dialogue'])
    question = encode_sentence(mode='question', sentence=question, token2id=t2i, maxlen=Config.max_len['question'])
    report=encode_sentence(mode='tar_report_input',sentence='',token2id=t2i,maxlen=Config.max_len['report'])
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
        dialogue_padding_mask, combined_mask, question_padding_mask = create_mask(dialogue_encoder, question_decoder,
                                                                                  report_decoder)
        predictions = model([dialogue_encoder, report_decoder,question_decoder,
                                     False,
                                     dialogue_padding_mask,
                                     combined_mask,
                                     dialogue_padding_mask,
                                     question_padding_mask])
        prediction_idx=predictions[0]
        #找到最后一个要预测的id
        predicted_ids = prediction_idx[-1, :]  # (batch_size, 1, vocab_size)
        predicted_id=tf.argmax(predicted_ids)
        if predicted_id==t2i['<eos>']:
            return tf.squeeze(report_decoder, axis=0)
        print(report_decoder)
        print(i2t[predicted_id.numpy()])
        predicted_id=tf.cast(tf.expand_dims([predicted_id],0),dtype=tf.int32)


        report_decoder=tf.concat([report_decoder, predicted_id], axis=-1)
    return tf.squeeze(report_decoder, axis=0)
if __name__ == '__main__':
    dialogue='技师说：你好，根据你描述的这种现象，应该是节气门太脏，或者是发动机积碳问题导致的，建议你去修理厂清洗一下节气门和进气道积碳问题，应该就可以解决。|车主说：节气门年前刚洗的，换的新空滤|技师说：主要问题就是怠速低|车主说：怎么解决大师|技师说：还是节气门这块没有清理干净或者是进气道积碳|车主说：清洗进气道积碳多少钱|技师说：100-150|车主说：清洗节气门多少钱|技师说：50|车主说：发动机有滋滋声正常吗|技师说：检查皮带|车主说：皮带要换门|车主说：皮带要换吗|技师说：这个需要检查一下才能确定|车主说：好谢啦'
    question='大师好。07款福克斯两厢今天开的好好的熄火再启动就熄火，要带点油门，才可以'
    report='根据你描述的这种现象，应该是节气门太脏，或者是发动机积碳问题导致的，建议你去修理厂清洗一下节气门和进气道积碳问题，应该就可以解决。'
    ids=predict(dialogue,question,report).numpy()
    print(ids)
    k=[i2t[i] for i in ids]
    print(''.join(k))