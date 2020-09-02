# -*- coding: utf-8 -*-
# @Time : 2020/8/18 上午10:15
# @Author : chezhonghao
# @description : 
# @File : TransformerModel.py
# @Software: PyCharm
import tensorflow.keras as keras
import tensorflow as tf
from base_transformer_model.component import positional_encoding
from base_transformer_model.custom_layers import EncoderLayer,DecoderLayer
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding,rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        # 将嵌入和位置编码相加。
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding,rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.pos_encoding2 = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        'look_ahead_mask, encoder_padding_mask, question_padding'
    def call(self, x,x2,enc_output, training,
             look_ahead_mask, encoder_padding_mask,question_padding):
        seq_len = tf.shape(x)[1]
        seq_len2=tf.shape(x2)[1]
        # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x2 *= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x2+=self.pos_encoding[:,:seq_len2,:]
        x = self.dropout(x,training)
        x2=self.dropout2(x2,training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x,x2, enc_output, training,
                                                   look_ahead_mask, encoder_padding_mask,
                                                   question_padding)
            # attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            # attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        # x.shape == (batch_size, target_seq_len, d_model)
        return x
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.embedding=tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input,rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target,rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size,activation='softmax')
    train_step_signature = [
        [
            tf.TensorSpec(shape=[None, None], dtype=tf.int32,name='inp'),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32,name='tar'),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='tar2'),
            tf.TensorSpec(shape=[],dtype=bool,name='training'),
            tf.TensorSpec(shape=[None,None,None,None],dtype=tf.float32,name='enc_padding_mask'),
            tf.TensorSpec(shape=[None, None, None, None],dtype=tf.float32, name='look_ahead_mask'),
            tf.TensorSpec(shape=[None, None, None, None],dtype=tf.float32, name='encoder_padding_mask'),
            tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32, name='question_padding'),
        ]
    ]
    @tf.function(input_signature=train_step_signature)
    def call(self, parameter_list):
        'inp,tar,tar2,training,enc_padding_mask,look_ahead_mask,encoder_padding_mask,question_padding'
        inp=parameter_list[0]
        tar=parameter_list[1]
        tar2=parameter_list[2]
        training=parameter_list[3]
        enc_padding_mask=parameter_list[4]
        look_ahead_mask=parameter_list[5]
        encoder_padding_mask=parameter_list[6]
        question_padding=parameter_list[7]
        inp=self.embedding(inp)
        tar=self.embedding(tar)
        tar2=self.embedding(tar2)
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder(
            tar,tar2,enc_output, training, look_ahead_mask, encoder_padding_mask,question_padding)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output

if __name__ == '__main__':
    # 测试transformer千万别删 这个
    from base_transformer_model.component import create_padding_mask, create_look_ahead_mask
    def create_masks(dialogue, question, report):
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
    dialogue = tf.constant([[1, 2, 3, 0, 0], [2, 0, 0, 0, 0]], dtype=tf.int32)
    question = tf.constant([[3, 4, 0, ], [2, 2, 0, ]], dtype=tf.int32)
    report = tf.constant([[3, 3, 1, 0], [1, 1, 0, 0]], dtype=tf.int32)
    dialogue_padding_mask, combined_mask, question_padding_mask = create_masks(dialogue, question, report)
    transformer=Transformer(num_layers=1, d_model=2, num_heads=1, dff=2, input_vocab_size=5,
                 target_vocab_size=5, pe_input=1000, pe_target=200, rate=0.1)
    kk=transformer(inp=dialogue,tar=report,tar2=question,training=False,enc_padding_mask=dialogue_padding_mask,
                   look_ahead_mask=combined_mask,encoder_padding_mask=dialogue_padding_mask,
                   question_padding=question_padding_mask)