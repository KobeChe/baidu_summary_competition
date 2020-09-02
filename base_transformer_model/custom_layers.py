# -*- coding: utf-8 -*-
# @Time : 2020/8/10 下午4:59
# @Author : chezhonghao
# @description : 
# @File : custom_layers.py
# @Software: PyCharm
import tensorflow as tf
from base_transformer_model.component import MultiHeadAttention,point_wise_feed_forward_network
import tensorflow.keras as keras


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        #decoder target的mha
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        #encoder到decoder的mha
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        #question的mha  question自己的mha
        self.mha_question=MultiHeadAttention(d_model,num_heads)
        #dialogue到question的mha   question看过dialogue的mha
        self.mha_question_dialogue=MultiHeadAttention(d_model,num_heads)
        #report看过question的attention
        self.mha_report_d_q=MultiHeadAttention(d_model,num_heads)
        #report带着question去看dialogue的mha
        self.mha_report_q_d_final=MultiHeadAttention(d_model,num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_question = tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                                                     name='layernorm_question')
        self.layernorm_question_dialogue=tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                                                            name='layernorm_question_dialogue')
        self.layernorm_report_q_d = tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                                                              name='layernorm_report_q_d')
        self.layernorm_report_q_d_final=tf.keras.layers.LayerNormalization(epsilon=1e-6,
                                                                           name='layernorm_report_q_d_final')

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.dropout_question=tf.keras.layers.Dropout(rate,
                                                      name='dropout_question')
        self.dropout_question_dialogue=tf.keras.layers.Dropout(rate,
                                                               name='dropout_question_dialogue')
        self.dropout_report_q_d=tf.keras.layers.Dropout(rate,
                                                        name='dropout_report_q_d')
        self.dropout_report_q_d_final=tf.keras.layers.Dropout(rate,
                                                              name='layernorm_report_q_d_final')

    def call(self, x,x2, enc_output, training,
             look_ahead_mask, encoder_padding_mask,question_padding):
        '''
        :param x: report ......<eos>
        :param x2: question
        :param enc_output:
        :param training:
        :param look_ahead_mask: 遮挡要预测的序列
        :param encoder_padding_mask: 遮挡encoder output
        :param question_padding:遮挡question序列
        :return:
        '''
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        #首先对question即x2进行mha
        attention_question,_=self.mha_question(x2,x2,x2,question_padding)
        attention_question=self.dropout_question(attention_question,training)
        out_question=self.layernorm_question(x2+attention_question)
        #question对dialogue的attention  此处不需要lookahead遮挡  question看过一次dialogue
        attention_q_d,_=self.mha_question_dialogue(enc_output,enc_output,out_question,encoder_padding_mask)
        attention_q_d=self.dropout_question_dialogue(attention_q_d,training)
        out_question_dialogue=self.layernorm_question_dialogue(out_question+attention_q_d)
        # print(encoder_padding_mask.shape)
        #report自己的attention
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training)
        out1 = self.layernorm1(attn1 + x)
        # print(out1.shape)
        # print(out_question_dialogue.shape)
        # print(encoder_padding_mask.shape)
        #report 到被question过滤的dialogue的attention    report看过question(已经看过dialogue的question)
        atten_report_d_q,_=self.mha_report_d_q(out_question_dialogue,out_question_dialogue,out1,question_padding)
        atten_report_d_q=self.dropout_report_q_d(atten_report_d_q,training)
        out_atten_report_d_q=self.layernorm_report_q_d(out1+atten_report_d_q)
        #看过question的report再去看dialogue
        attention_report_final,_=self.mha_report_q_d_final(enc_output,enc_output,out_atten_report_d_q,encoder_padding_mask)
        attention_report_final=self.dropout_report_q_d_final(attention_report_final)
        attention_report_final_out=self.layernorm_report_q_d_final(out_atten_report_d_q+attention_report_final)


        ffn_output = self.ffn(attention_report_final_out)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + attention_report_final_out)  # (batch_size, target_seq_len, d_model)
        return out3

if __name__ == '__main__':
    #测试decoder layer
    decoder=DecoderLayer(d_model=2, num_heads=1, dff=3, rate=0.1)
    # encoder=EncoderLayer(d_model=2, num_heads=1, dff=3, rate=0.1)
    # dialogue=tf.constant([[1,2,3,0,0],[2,0,0,0,0]],dtype=tf.int32)
    # question=tf.constant([[3,4,0,],[2,2,0,]],dtype=tf.int32)
    # report=tf.constant([[3,3,1,0],[1,1,0,0]],dtype=tf.int32)
    # embedding=tf.keras.layers.Embedding(5,2)
    # dialogue_embedding = embedding(dialogue)
    # question_embedding = embedding(question)
    # report_embedding = embedding(report)
    # dialogue_padding_mask, combined_mask, question_padding_mask=create_masks(dialogue,question,report)
    # encoder_output=encoder(dialogue_embedding, False, dialogue_padding_mask)
    # decoder_output=decoder(report_embedding,question_embedding, encoder_output, False,
    #          combined_mask, dialogue_padding_mask,question_padding_mask)

