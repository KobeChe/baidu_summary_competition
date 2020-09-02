# -*- coding: utf-8 -*-
# @Time : 2020/8/24 下午9:15
# @Author : chezhonghao
# @description : 
# @File : utils.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
	return pos * angle_rates
def positional_encoding(position, d_model):
	angle_rads = get_angles(np.arange(position)[:, np.newaxis],
							np.arange(d_model)[np.newaxis, :],
							d_model)
	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
	pos_encoding = angle_rads[np.newaxis, ...]
	return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
	seq = tf.cast(tf.math.equal(seq, 1), tf.float32)
	# add extra dimensions to add the padding
	# to the attention logits.
	return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
def create_look_ahead_mask(size):
	mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
	return mask  # (seq_len, seq_len)
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
    return tf.cast(dialogue_padding_mask,tf.int32), tf.cast(combined_mask,tf.int32), tf.cast(question_padding_mask,tf.int32)
def scaled_dot_product_attention(q, k, v, mask):
	"""Calculate the attention weights.
	q, k, v must have matching leading dimensions.
	k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
	The mask has different shapes depending on its type(padding or look ahead)
	but it must be broadcastable for addition.
	Args:
	q: query shape == (..., seq_len_q, depth)
	k: key shape == (..., seq_len_k, depth)
	v: value shape == (..., seq_len_v, depth_v)
	mask: Float tensor with shape broadcastable
	      to (..., seq_len_q, seq_len_k). Defaults to None.
	Returns:
	output, attention_weights
	"""
	matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
	# scale matmul_qk
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
	# add the mask to the scaled tensor.
	if mask is not None:
		scaled_attention_logits += (tf.cast(mask,scaled_attention_logits.dtype) * -1e9)
	# softmax is normalized on the last axis (seq_len_k) so that the scores
	# add up to 1.
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
	output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
	return output, attention_weights

def _calc_final_dist(ecoder_input,vocab_dists, attn_dists, p_gens):
	"""Calculate the final distribution, for the pointer-generator base_transformer_model
	Args:
	vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
	attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
	Returns:
	final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
	"""
	# Multiply vocab dists by p_gen and attention dists by (1-p_gen)
	vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
	attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]
	batch_nums = tf.range(0, limit=vocab_dists[0].shape[0])  # shape (batch_size)
	batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
	attn_len = tf.shape(attn_dists[0])[1]  # number of states we attend over
	batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
	indices = tf.stack((batch_nums, tf.transpose(ecoder_input,[1,0])), axis=2)  # shape (batch_size, enc_t, 2)
	shape = [vocab_dists[0].shape[0], vocab_dists[0].shape[1]]
	attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
							attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)
	final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
				   zip(vocab_dists, attn_dists_projected)]
	return final_dists
if __name__ == '__main__':
	a=tf.constant([1,2,3,4])
	b=tf.expand_dims(a,axis=-1)
	print(b)