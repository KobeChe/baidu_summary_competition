# -*- coding: utf-8 -*-
# @Time : 2020/8/24 下午9:19
# @Author : chezhonghao
# @description : 
# @File : layers.py
# @Software: PyCharm
import tensorflow as tf
from transformer_pointer_model_cross_attention.utils import positional_encoding, scaled_dot_product_attention
import tensornetwork as tn
from datetime import datetime
class Embedding(tf.keras.layers.Layer):
	def __init__(self, vocab_size, d_model):
		super(Embedding, self).__init__()
		self.vocab_size = vocab_size
		self.d_model = d_model
		self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
		self.pos_encoding = positional_encoding(vocab_size, d_model)
	def call(self, x):
		embed_x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
		embed_x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		embed_x += self.pos_encoding[:, :tf.shape(x)[1], :]
		return embed_x
class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model
		assert d_model % self.num_heads == 0
		self.depth = d_model // self.num_heads
		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)
		self.dense = tf.keras.layers.Dense(d_model)
	def split_heads(self, x, batch_size):
		'''
		分割八个头，用于计算attention
		:param x:
		:param batch_size:
		:return:
		'''
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])
	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]
		q = self.wq(q)  # (batch_size, seq_len, d_model)
		k = self.wk(k)  # (batch_size, seq_len, d_model)
		v = self.wv(v)  # (batch_size, seq_len, d_model)
		q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
		scaled_attention, attention_weights = scaled_dot_product_attention(
			q, k, v, mask)
		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
		concat_attention = tf.reshape(scaled_attention,
										(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
		output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
		return output, attention_weights
def point_wise_feed_forward_network(d_model, dff):
	'''
	加速的feed_forward,此处将dff默认1024，如果需要修改，则需要修改加速代码
	:param d_model:
	:param dff:
	:return:
	'''
	return tf.keras.Sequential([
		tf.keras.layers.Dense(dff),
		# point_wise_fn_accerator(),# (batch_size, seq_len, dff)
		tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
	])
#tensornetwork加速point_wise_feed_forward_network   dff=1024
class point_wise_fn_accerator(tf.keras.layers.Layer):
	'''
	加速point_wise_fn_accerator,成功缩减1千万参数
	'''
	def __init__(self):
		super(point_wise_fn_accerator, self).__init__()
		#分解[d_model,dff]权重矩阵的核心
		self.a_var=tf.Variable(tf.keras.initializers.GlorotUniform()(shape=[8,8,2]),name='a',trainable=True)
		self.b_var=tf.Variable(tf.keras.initializers.GlorotUniform()(shape=[8,8,2,2]),name='b',trainable=True)
		self.c_var=tf.Variable(tf.keras.initializers.GlorotUniform()(shape=[16,8,2]),name='c',trainable=True)
		self.bias=tf.Variable(tf.zeros(shape=[8,8,16]),name='bias',trainable=True)
	def call(self, inputs):
		'''
		定义张量边的链接以及contraction
		:param inputs:
		:return:
		'''
		def f(input_vec,a_var,b_var,c_var,bias):
			input_vec = tf.reshape(input_vec, shape=[8, 8, 8])
			x_node=tn.Node(input_vec,backend='tensorflow',name='x_node')
			a_node=tn.Node(tensor=a_var,backend='tensorflow',name='a_node')
			b_node=tn.Node(tensor=b_var,backend='tensorflow',name='b_node')
			c_node=tn.Node(tensor=c_var,backend='tensorflow',name='c_node')
			#将变量的边与输入的边相连
			a_node[1] ^ x_node[0]
			b_node[1] ^ x_node[1]
			c_node[1] ^ x_node[2]
			#将变量的边相连
			a_node[2] ^ b_node[2]
			b_node[3] ^ c_node[2]
			#contraction
			start=datetime.now()
			result=tn.contractors.auto([x_node,a_node,b_node,c_node],
									   output_edge_order=[a_node[0],b_node[0],c_node[0]]).tensor
			print(datetime.now()-start)
			# temp_node2=b_node @ temp_node
			# result=(c_node @ temp_node2).tensor
			return result+bias
		def confuse_dimation(input_vec_reduce_dim,a_var,b_var,c_var,bias):

			result=tf.vectorized_map(lambda input_v:f(input_v,a_var,b_var,c_var,bias),input_vec_reduce_dim)

			return result


		result=tf.vectorized_map(lambda input_v_reduce_dim:confuse_dimation(input_v_reduce_dim,
																			self.a_var,
																			self.b_var,
																			self.c_var,
																			self.bias),
								 inputs)

		return tf.nn.relu(tf.reshape(result,shape=[inputs.shape[0],-1,1024]))


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
		self.mha1 = MultiHeadAttention(d_model, num_heads)
		self.mha2 = MultiHeadAttention(d_model, num_heads)
		self.mha_question = MultiHeadAttention(d_model,num_heads)
		self.mha_question_report = MultiHeadAttention(d_model, num_heads)

		self.ffn = point_wise_feed_forward_network(d_model, dff)
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm_question = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm_question_report = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)
		self.dropout3 = tf.keras.layers.Dropout(rate)
		self.dropout_question = tf.keras.layers.Dropout(rate)
		self.dropout_question_report = tf.keras.layers.Dropout(rate)
	def call(self, x, enc_output,question, training, look_ahead_mask, padding_mask,question_padding):
		# enc_output.shape == (batch_size, input_seq_len, d_model)
		attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
		attn1 = self.dropout1(attn1, training=training)
		out1 = self.layernorm1(attn1 + x)
        #question自己attention
		attn_question, attn_weights_block_q = self.mha_question(question, question, question, question_padding)  # (batch_size, target_seq_len, d_model)
		attn_question = self.dropout_question(attn_question, training=training)
		out_question = self.layernorm_question(attn_question + question)

		#report对question进行attention
		attn_question_q_r, attn_weights_block_q_r = self.mha_question_report(out_question, out_question, out1,
																question_padding)  # (batch_size, target_seq_len, d_model)
		attn_question_q_r = self.dropout_question_report(attn_question_q_r, training=training)
		out1 = self.layernorm_question_report(attn_question_q_r + out1)
		attn2, attn_weights_block2 = self.mha2(
			enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
		attn2 = self.dropout2(attn2, training=training)
		out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
		# start = datetime.now()

		ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
		# print(datetime.now() - start)
		ffn_output = self.dropout3(ffn_output, training=training)
		out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
		return out3, attn_weights_block1, attn_weights_block2
