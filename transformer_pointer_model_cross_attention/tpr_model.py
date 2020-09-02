# -*- coding: utf-8 -*-
# @Time : 2020/8/24 下午9:15
# @Author : chezhonghao
# @description : 
# @File : tpr_model.py
# @Software: PyCharm
import os
# os.environ['CUDA_VISIBLE_DEVICES']="-1"
import tensorflow as tf
from transformer_pointer_model_cross_attention.layers import Embedding, EncoderLayer, DecoderLayer
from transformer_pointer_model_cross_attention.utils import _calc_final_dist
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x, training, mask):
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.depth = d_model // self.num_heads
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.dropout2=tf.keras.layers.Dropout(rate)
        self.Wh = tf.keras.layers.Dense(1)
        self.Ws = tf.keras.layers.Dense(1)
        self.Wx = tf.keras.layers.Dense(1)
        self.V = tf.keras.layers.Dense(1)

    def call(self, embed_x, enc_output,question, training, look_ahead_mask, padding_mask,question_padding):
        attention_weights = {}
        out = self.dropout(embed_x,training)
        question==self.dropout(question,training)
        for i in range(self.num_layers):
            out, block1, block2 = self.dec_layers[i](out, enc_output, question,training,
                                                     look_ahead_mask, padding_mask,question_padding)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        # out.shape == (batch_size, target_seq_len, d_model)
        # context vectors
        enc_out_shape = tf.shape(enc_output)
        context = tf.reshape(enc_output, (enc_out_shape[0], enc_out_shape[1], self.num_heads,
                                          self.depth))  # shape : (batch_size, input_seq_len, num_heads, depth)
        context = tf.transpose(context, [0, 2, 1, 3])  # (batch_size, num_heads, input_seq_len, depth)
        context = tf.expand_dims(context, axis=2)  # (batch_size, num_heads, 1, input_seq_len, depth)
        attn = tf.expand_dims(block2, axis=-1)  # (batch_size, num_heads, target_seq_len, input_seq_len, 1)
        context = context * attn  # (batch_size, num_heads, target_seq_len, input_seq_len, depth)
        context = tf.reduce_sum(context, axis=3)  # (batch_size, num_heads, target_seq_len, depth)
        context = tf.transpose(context, [0, 2, 1, 3])  # (batch_size, target_seq_len, num_heads, depth)
        context = tf.reshape(context,
                             (tf.shape(context)[0], tf.shape(context)[1], self.d_model))  # (batch_size, target_seq_len, d_model)
        # P_gens computing
        a = self.Wx(embed_x)
        b = self.Ws(out)
        c = self.Wh(context)
        p_gens = tf.sigmoid(self.V(a + b + c))
        return out, attention_weights, p_gens
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, batch_size, rate=0.1):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.model_depth = d_model
        self.num_heads = num_heads
        self.embedding = Embedding(vocab_size, d_model)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, rate)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    train_step_signature = [
        [
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='inp'),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='question'),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='tar'),
            tf.TensorSpec(shape=[], dtype=bool, name='training'),
            tf.TensorSpec(shape=[None, None, None, None], dtype=tf.int32, name='enc_padding_mask'),
            tf.TensorSpec(shape=[None, None, None, None], dtype=tf.int32, name='look_ahead_mask'),
            tf.TensorSpec(shape=[None, None, None, None], dtype=tf.int32, name='dec_padding_mask'),
            tf.TensorSpec(shape=[None, None, None, None], dtype=tf.int32, name='question_padding'),
        ]
    ]
    # @tf.function(input_signature=train_step_signature)
    def call(self, inp,question,tar,training,enc_padding_mask,look_ahead_mask,dec_padding_mask,question_padding):
        'inp,question,tar,training,enc_padding_mask,look_ahead_mask,dec_padding_mask,question_padding'
        # inp=paramlist[0]
        # question=paramlist[1]
        # tar=paramlist[2]
        # training=paramlist[3]
        # enc_padding_mask=paramlist[4]
        # look_ahead_mask=paramlist[5]
        # dec_padding_mask=paramlist[6]
        # question_padding=paramlist[7]
        embed_x = self.embedding(inp)
        embed_dec = self.embedding(tar)
        embed_question=self.embedding(question)
        enc_output = self.encoder(embed_x, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights, p_gens = self.decoder(embed_dec,enc_output, embed_question,training, look_ahead_mask,
                                                             dec_padding_mask,question_padding)
        output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        output = tf.nn.softmax(output)  # (batch_size, tar_seq_len, vocab_size)
        # output = tf.concat([output, tf.zeros((tf.shape(output)[0], tf.shape(output)[1], max_oov_len))], axis=-1) # (batch_size, targ_seq_len, vocab_size+max_oov_len)
        attn_dists = attention_weights[
            'decoder_layer{}_block2'.format(self.num_layers)]  # (batch_size,num_heads, targ_seq_len, inp_seq_len)
        attn_dists = tf.reduce_sum(attn_dists, axis=1) / self.num_heads  # (batch_size, targ_seq_len, inp_seq_len)

        final_dists = _calc_final_dist(tf.unstack(inp,axis=1),tf.unstack(output,axis=1), tf.unstack(attn_dists, axis=1),
                                       tf.unstack(p_gens,axis=1))
        final_output = tf.stack(final_dists, axis=1)
        return final_output, attention_weights
if __name__ == '__main__':
    from datetime import datetime
    transformer=Transformer(num_layers=6, d_model=512, num_heads=8, dff=1024, vocab_size=1000, batch_size=10, rate=0.1)
    inp=tf.constant([[1,2,3,0],[2,3,4,0]])
    tar=tf.constant([[2,3,4],[2,3,0]])
    question=tf.constant([[2,3,4],[2,3,0]])
    start=datetime.now()
    transformer(inp, question,tar, False, None, None, None,None)
    print(datetime.now()-start)
    start = datetime.now()
    transformer(inp, question, tar, False, None, None, None, None)
    print(datetime.now() - start)
    start = datetime.now()
    transformer(inp, question, tar, False, None, None, None, None)
    print(datetime.now() - start)
    # print(tf.split(inp,num_or_size_splits=4,axis=1))
    # print(tf.unstack(inp,axis=1))
    transformer.summary()