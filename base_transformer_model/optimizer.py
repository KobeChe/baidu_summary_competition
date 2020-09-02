# -*- coding: utf-8 -*-
# @Time : 2020/8/18 下午4:58
# @Author : chezhonghao
# @description : 
# @File : optimizer.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
if __name__ == '__main__':
    # learning_rate = CustomSchedule(512)
    # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
    #                                      epsilon=1e-9)
    temp_learning_rate_schedule = CustomSchedule(512)

    plt.plot(temp_learning_rate_schedule(tf.range(8000000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()