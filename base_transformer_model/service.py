# -*- coding: utf-8 -*-
# @Time : 2020/8/18 下午5:20
# @Author : chezhonghao
# @description : 
# @File : service.py
# @Software: PyCharm
from base_transformer_model.optimizer import CustomSchedule

from tqdm import tqdm
from base_transformer_model.TransformerModel import Transformer
from base_transformer_model.component import create_padding_mask,create_look_ahead_mask
from preprocess import tfrecord_reader,encode_sentence,load_vocab
from tensorflow import keras
import tensorflow as tf
import time
from base_transformer_model.config import Config
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
t2i, i2t = load_vocab(Config.vocab_path)
saved_model_path='saved_model_trasformer'
num_layers = Config.num_layers
d_model = Config.d_model
dff = Config.dff
num_heads = Config.num_heads
input_vocab_size = Config.vocab_size
target_vocab_size = Config.vocab_size
dropout_rate = 0.1


# train_step_signature = [
#     tf.TensorSpec(shape=[None, None], dtype=tf.int32),
#     tf.TensorSpec(shape=[None, None], dtype=tf.int32),
# ]
def loss_function(real,pred):
    loss_object=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1,reduction='none')
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
# transformer=Transformer(num_layers=1, d_model=2, num_heads=1, dff=2, input_vocab_size=5,
#                  target_vocab_size=5, pe_input=1000, pe_target=200, rate=0.1)
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)
# transformer_model=tf.saved_model.load(Config.saved_model_path)
learning_rate = CustomSchedule(Config.d_model)
optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                  epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy= tf.keras.metrics.Mean(name='train_accuracy')
train_accuracy_per = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_per')
# def accuracy_rate(real,pred):
#     pred = pred[:, Config.max_len['question']:, :]
#     #把0都换成false
#     return train_accuracy_per(real,pred)
# @tf.function(input_signature=train_step_signature)
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, Config.ckpt, max_to_keep=1)
# 如果检查点存在，则恢复最新的检查点。
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
def train_step(dialogue_encoder,question_decoder,report_decoder,report_real):
    '''
    :param inp: [dialogue]
    :param tar: [question,report]
    :return:
    '''
    #tar_input没有eos但是有bos
    #real没有bos但是有eos
    # tar_input=tf.concat([question,tar_report_input],axis=-1)
    # tar_input_test=tar_input[0].numpy()
    # print('tar_input_test_train:',''.join([i2t[idx] for idx in tar_input_test]))
    # tar_inp = tar[:, :-1]
    # tar_inp=tar
    #应该输出的真实
    # tar_real = tar_report_output
    dialogue_padding_mask, combined_mask, question_padding_mask = create_mask(dialogue_encoder, question_decoder, report_decoder)
    with tf.GradientTape() as tape:
        'inp,tar,tar2,training,enc_padding_mask,look_ahead_mask,encoder_padding_mask,question_padding'
        predictions= transformer([dialogue_encoder, report_decoder,question_decoder,
                                     True,
                                     dialogue_padding_mask,
                                     combined_mask,
                                     dialogue_padding_mask,
                                     question_padding_mask])
        loss = loss_function(report_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
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
        print('one_hot_true_eval_test_final',''.join([i2t[idx] for idx in one_hot_true_eval_test]))
        pred_eval_test_final = tf.argmax(pred, axis=-1)[0].numpy()
        print(pred_eval_test_final.shape)
        print('pred_eval_test_final', ''.join([i2t[idx] for idx in pred_eval_test_final]))
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
        tarinput_test=dialogue_encoder[0].numpy()
        print('tar_input_eval:',''.join([i2t[idx] for idx in tarinput_test]))
        tar_output_test = report_real[0].numpy()
        print('tar_output_test_eval:', ''.join([i2t[idx] for idx in tar_output_test]))
        dialogue_padding_mask, combined_mask, question_padding_mask = create_mask(dialogue_encoder, question_decoder, report_decoder)
        # print(tf.reduce_sum(combined_mask[0][0][0]))
        # print(tf.reduce_sum(combined_mask[0][0][145]))
        predictions = transformer([dialogue_encoder, report_decoder, question_decoder,
                                  False,
                                  dialogue_padding_mask,
                                  combined_mask,
                                  dialogue_padding_mask,
                                  question_padding_mask])
        loss=eval_loss_function(report_real,predictions)
        eval_loss(loss)
    print('eval loss : ',eval_loss.result().numpy())
    return eval_loss.result().numpy()
def train():
    data_batch = tfrecord_reader(Config.batch_size, Config.serilized_path, Config.max_len,
                                 n_readers=2, buffer_size=1000, num_parallel=2)
    min_eval_loss = 100000
    for epoch in range(Config.epoches):
        train_loss.reset_states()
        train_accuracy.reset_states()
        for batch,data in tqdm(enumerate(data_batch)):
            dialogue_encoder=data['dialogue']
            question_decoder=data['question']
            report_decoder=data['tar_report_input']
            report_real=data['tar_report_output']
            # print(tar_report_input[0])
            # print(question[0])
            train_step(dialogue_encoder,question_decoder,report_decoder,report_real)
            if batch % 1000 == 0:
                current_loss_eval=evaluate()
                logger.info('Epoch {} Batch {} Train_Loss {:.4f}  eval_loss {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(),current_loss_eval))
                if current_loss_eval<min_eval_loss:
                    min_eval_loss=current_loss_eval
                    tf.saved_model.save(transformer, Config.saved_model_path)
                    ckpt_save_path = ckpt_manager.save()
            # if batch>Config.steps_per_epoch:
            #     break
        logger.info('Epoch {} Train_Loss {:.4f} '.format(epoch + 1,
                                                            train_loss.result(),
                                                             ))

if __name__ == '__main__':
    # transformer_model = SummaryTransformer(num_layers=6, num_heads=8, d_model=512, dff=1024, vocab_size=4000,
    #                                        pe_encoder_input=100, pe_decoder_input=50)
    # x=tf.constant([0,1,2,2],dtype=tf.int32)
    # print(tf.math.logical_not(tf.equal(x,0)))
    train()
    # predict('07款大众菠萝1.4正时链条怎么对','众菠萝1.4正时链条')
    # evaluate()

