# -*- coding: utf-8 -*-
# @Time : 2020/8/28 上午10:39
# @Author : chezhonghao
# @description : 
# @File : training_utils.py
# @Software: PyCharm
import tensorflow as tf
from transformer_pointer_model_cross_attention.utils import create_masks
from base_transformer_model.config import Config
from preprocess import load_vocab,tfrecord_reader
from transformer_pointer_model_cross_attention.tpr_model import Transformer
from tqdm import tqdm
import time
from preprocess import load_vocab
import logging
logger = logging.getLogger('test')
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
file_handler = logging.FileHandler('test2.log')
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
model=Transformer(num_layers=Config.num_layers, d_model=Config.d_model, num_heads=Config.num_heads, dff=Config.dff,
                  vocab_size=Config.vocab_size, batch_size=Config.batch_size,
                  rate=0.1)
t2i,i2t=load_vocab(Config.vocab_path)
def evaluate():
    data_batch = tfrecord_reader(Config.eval_batch_size, Config.serilized_path_eval, Config.max_len,
                                 n_readers=3, buffer_size=100, num_parallel=3)
    eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    eval_loss.reset_states()
    def eval_loss_function(real, pred):
        loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1, reduction='none')
        # (batch_size,seq_len)
        # (batch_size,seq_len,vocab_size)
        real_test=real[0].numpy()
        pred_test=tf.argmax(pred,axis=-1)[0].numpy()
        print('real_test_eval',''.join([i2t[idx] for idx in real_test]))
        print('pred_test_eval',''.join([i2t[idx] for idx in pred_test]))
        # 把real为0的部分标记成False
        mask = tf.math.logical_not(tf.equal(real, 0))
        one_hot_true = tf.one_hot(real, depth=Config.vocab_size)
        pred = pred
        # pred_final_test=tf.argmax(pred,axis=-1).numpy()
        one_hot_true_eval_test=tf.argmax(one_hot_true,axis=-1)[0].numpy()
        # print(one_hot_true.shape)
        # print('one_hot_true[0]',one_hot_true[0].numpy())
        # print('pred[0]',pred[0].numpy())
        loss_ = loss_object(one_hot_true, pred)
        # print('loss_.numpy()',loss_.numpy())
        mask = tf.cast(mask, loss_.dtype)
        # print('mask.numpy()',mask.numpy())
        loss_ *= mask
        # print('loss_.numpy()',loss_.numpy())
        loss_ = tf.reduce_sum(loss_)
        # print('loss_.numpy()_final',loss_.numpy())
        sig_num = tf.reduce_sum(mask)
        # print('sig_num.numpy',sig_num.numpy)
        return loss_ / sig_num
    for batch, data in enumerate(data_batch):
        dialogue_encoder = data['dialogue']
        report_decoder=data['tar_report_input']
        report_real=data['tar_report_output']
        question_decoder = data['question']
        # tarinput_test=dialogue_encoder[0].numpy()
        # print('tar_input_eval:',''.join([i2t[idx] for idx in tarinput_test]))
        tar_output_test = report_real[0].numpy()
        # print('tar_output_test_eval:', ''.join([i2t[idx] for idx in tar_output_test]))
        dialogue_padding_mask, combined_mask, question_padding_mask = create_masks(dialogue_encoder, question_decoder, report_decoder)
        # print(tf.reduce_sum(combined_mask[0][0][0]))
        # print(tf.reduce_sum(combined_mask[0][0][145]))
        predictions, attn_weights = model(dialogue_encoder, question_decoder,
                                      report_decoder, True,
                                      dialogue_padding_mask,
                                      combined_mask,
                                      dialogue_padding_mask,
                                      question_padding_mask)
        loss=eval_loss_function(report_real,predictions)
        eval_loss(loss)
    print('eval loss : ',eval_loss.result().numpy())
    return eval_loss.result().numpy()

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

def loss_function(loss_object,real,pred):
    #(batch_size,seq_len)
    #(batch_size,seq_len,vocab_size)
    # real_test=real[0].numpy()
    # pred_test=tf.argmax(pred,axis=-1)[0].numpy()
    # print('real_test_train',''.join([i2t[idx] for idx in real_test]))
    # print('pred_test_train',''.join([i2t[idx] for idx in pred_test]))
    #把real为0的部分标记成False
    mask=tf.math.logical_not(tf.equal(real,0))
    one_hot_true=tf.one_hot(real,depth=Config.vocab_size)
    pred=pred
    # pred_final_test=tf.argmax(pred,axis=-1).numpy()
    loss_=loss_object(one_hot_true,pred)
    mask=tf.cast(mask, loss_.dtype)
    loss_*=mask
    loss_=tf.reduce_sum(loss_)
    sig_num=tf.reduce_sum(mask)
    return loss_ / sig_num
def train_step(data,model,optimizer, loss_object, train_loss_metric):
    dialogue_encoder = data['dialogue']
    report_decoder = data['tar_report_input']
    report_real = data['tar_report_output']
    question_decoder = data['question']
    dialogue_padding_mask, combined_mask, question_padding_mask = create_masks(dialogue_encoder, question_decoder,
                                                                              report_decoder)

    with tf.GradientTape() as tape:
        output, attn_weights = model(dialogue_encoder,question_decoder,
                                     report_decoder, True,
                                     dialogue_padding_mask,
                                     combined_mask,
                                     dialogue_padding_mask,
                                     question_padding_mask)
        loss = loss_function(loss_object, report_real, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_metric(loss)
learning_rate = CustomSchedule(Config.d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
ckpt = tf.train.Checkpoint(transformer=model,
                               optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, Config.ckpt, max_to_keep=1)
ckpt_manager_train = tf.train.CheckpointManager(ckpt, Config.train_ckpt, max_to_keep=1)

# 如果检查点存在，则恢复最新的检查点。
# if ckpt_manager_train.latest_checkpoint:
#     ckpt.restore(ckpt_manager_train.latest_checkpoint)
#     print(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')
def train_model():
    min_eval_loss = 34.2168
    min_train_loss=34.6679
    data_batch = tfrecord_reader(Config.batch_size, Config.serilized_path, Config.max_len,
                                 n_readers=2, buffer_size=1000, num_parallel=2)
    loss_object = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1, reduction='none')
    train_loss_metric = tf.keras.metrics.Mean(name="train_loss_metric")
    for epoch in range(Config.epoches):
        train_loss_metric.reset_states()
        for batch, data in tqdm(enumerate(data_batch)):
            train_step(data,model,optimizer,loss_object,train_loss_metric)
        # current_loss_eval = evaluate()
        # logger.info('Epoch {}  Train_Loss {:.4f}  eval_loss {:.4f}'.format(
        #     epoch + 1,  train_loss_metric.result(), current_loss_eval))
        # if current_loss_eval < min_eval_loss:
        #     min_eval_loss = current_loss_eval
        #     # tf.saved_model.save(base_transformer_model, Config.saved_model_path)
        #     ckpt_save_path = ckpt_manager.save()
        #     base_transformer_model.save_weights('../transformer_saved_weights/')
        # if train_loss_metric.result()<min_train_loss:
        #     min_train_loss=train_loss_metric.result()
        #     train_ckpt_save_path = ckpt_manager_train.save()
        #     base_transformer_model.save_weights('../train_transformer_saved_weights/')
    #     logger.info('Epoch {} Train_Loss {:.4f} '.format(epoch + 1,
    #                                                      train_loss_metric.result(),
    #                                                      ))

if __name__ == '__main__':
    train_model()