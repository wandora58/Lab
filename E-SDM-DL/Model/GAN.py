
import csv
import os
import pathlib
import tensorflow as tf
import logging
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

import math
import numpy as np
import pandas as pd

from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Activation, PReLU, ReLU
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCH_COLUMN = 'EPOCH'
LOSS_COLUMN = 'LOSS'
ACC_COLUMN = 'ACC'

class GAN:
    def __init__(self, max_epoch, batch_size, sample, test, user, total_bit, SNR, input_type):
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.sample = sample
        self.test = test
        self.user = user
        self.total_bit = total_bit
        self.SNR = SNR
        self.input_type = input_type
        self.gen_weight, self.dis_weight = self.create_weight()


    def set_trainable(self, model, trainable=False):
        model.trainable = trainable
        try:
            layers = model.layers
        except:
            return
        for layer in layers:
            self.set_trainable(layer, trainable)


    def generator(self):
        input = Input(shape=(self.input_len,))

        x = Dense(units=self.input_len*4)(input)
        x = PReLU()(x)
        # x = ReLU()(x)
        # x = BentLinear()(x)
        # x = Activation('linear')(x)

        x = Dense(units=self.input_len*2)(x)
        x = PReLU()(x)
        # x = ReLU()(x)
        # x = BentLinear()(x)
        # x = Activation('linear')(x)

        output = Dense(units=self.ans_len, activation='softmax')(x)

        return Model(inputs=input, outputs=output, name='generator')


    def discriminator(self):
        input = Input(shape=(self.ans_len,))

        x = Dense(units=64)(input)
        x = ReLU()(x)

        x = Dense(units=32)(x)
        x = ReLU()(x)

        output = Dense(units=self.ans_len, activation='softmax')(x)

        return Model(inputs=input, outputs=output, name='discriminator')


    def create_model(self, input_len, ans_len):
        self.input_len = input_len
        self.ans_len = ans_len
        input = Input(shape=(self.input_len,))
        answer = Input(shape=(self.ans_len,))

        gen = self.generator()
        gen_output = gen(input)

        dis = self.discriminator()
        dis_output = dis(gen_output)
        dis_answer = dis(answer)

        self.set_trainable(gen, True)
        self.set_trainable(dis, False)
        gen_train_stage = Model(inputs=[input, answer], outputs=[gen_output, dis_output], name='gen_train_stage')
        gen_train_optimizer = Adam(beta_1=0.9)
        gen_train_stage.compile(optimizer=gen_train_optimizer,
                                loss={'generator': 'categorical_crossentropy', 'discriminator': 'categorical_crossentropy'},
                                loss_weights={'generator': self.gen_weight, 'discriminator': self.dis_weight})

        self.set_trainable(gen, False)
        self.set_trainable(dis, True)
        dis_train_stage = Model(inputs=[input, answer], outputs=[dis_output, dis_answer], name='dis_train_stage')
        dis_train_optimizer = Adam(beta_1=0.9)
        dis_train_stage.compile(optimizer=dis_train_optimizer, loss='categorical_crossentropy',
                                metrics=['categorical_accuracy'])

        self.set_trainable(gen, True)
        self.set_trainable(dis, False)
        gen_test_stage = Model(inputs=input, outputs=gen_output, name='gen_test_stage')
        gen_test_optimizer = Adam(lr=0.0001, beta_1=0.9)
        gen_test_stage.compile(optimizer=gen_test_optimizer, loss='categorical_crossentropy')

        return gen_train_stage, dis_train_stage, gen_test_stage


    def evaluate(self, epoch, input_test, answer_test, gen_test_stage, gen_score):
        self.gen_weight, self.dis_weight = self.create_weight()

        target = np.zeros(self.test, dtype=np.int)
        target_pred = np.zeros(self.test, dtype=np.int)

        for i in range(self.test):
            input = np.zeros((1, self.input_len), dtype=np.float)
            answer = list(answer_test[i,:])

            for j in range(self.input_len):
                input[0, j] = input_test[i, j]

            predict = gen_test_stage.predict(input, batch_size=1)
            predict = [int(j) for j in predict[0]]

            target[i] = answer.index(1)
            target_pred[i] = predict.index(1)


        df_pred = pd.DataFrame({'target':target, 'target_pred':target_pred})
        loss = gen_score[0]
        acc = accuracy_score(target, target_pred)

        self.file_writing(epoch, df_pred, loss, acc)


    def file_writing(self, epoch, df_pred, loss, acc):

        file = "Data/epoch_data/{}/user{}_bit{}_SNR={}.csv".format(self.input_type, self.user, self.total_bit, self.SNR)
        if not os.path.exists(file):
            pathlib.Path(file).touch()

        columns = [EPOCH_COLUMN, LOSS_COLUMN, ACC_COLUMN]

        if epoch == 0:
            with open(file, 'w+') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

        with open(file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow({
                EPOCH_COLUMN: epoch,
                LOSS_COLUMN: loss,
                ACC_COLUMN: acc
            })

        print('--------------------------------')
        print('TYPE {}'.format(self.input_type))
        print('EPOCH: {}'.format(epoch))
        print('LOSS: {}'.format(loss))
        print('ACC: {}%'.format(acc))
        print(df_pred)
        print('--------------------------------')


    def illustrate(self):
        epoch = np.arange(self.max_epoch, dtype=np.float)
        x = np.zeros(self.max_epoch, dtype=np.float)

        file = "Data/epoch_data/{}/{}/size{}_pilot{}_user{}.csv".format(self.pilot_type, self.input_type, self.size, self.pilot_len, self.user)
        with open(file, 'r') as f:
            h = next(csv.reader(f))
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            i = 0
            for row in reader:
                epoch[i] = row[0]
                x[i] = row[1]
                i += 1

        # figure　
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(epoch, x, linestyle='-', color='g', label="{}_{}".format(self.pilot_type, self.input_type))

        # y軸の範囲設定
        # ax.set_ylim([-0.1,1.1])

        # ひげ消す
        plt.gca().xaxis.set_tick_params(direction='in')
        plt.gca().yaxis.set_tick_params(direction='in')

        # x軸間隔
        # plt.xticks([0,5,10,15,20])

        # y軸間隔
        plt.yticks([0, -5, -10, -15, -20, -25, -28])

        # x軸,y軸のラベル付け
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Normalized MSE [dB]', fontsize=12)

        # グリッド表示
        plt.grid(which="both")

        # 凡例とタイトル
        ax.legend(loc='best', prop={'size': 12})

        # 保存　
        plt.savefig('Data/image/{}/{}/user{}_plen{}.pdf'.format(self.pilot_type, self.input_type, self.user, self.pilot_len))


    def save_model(self, gen_test_stage):
        model_json_str = gen_test_stage.to_json()

        model_dir = "Data/model_data/{}/{}".format(self.pilot_type, self.input_type)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file = "{}/size{}_pilot{}_user{}.json".format(model_dir, self.size, self.pilot_len, self.user)
        with open(model_file, 'w') as f:
            f.write(model_json_str)


        weight_dir = "Data/model_weight/{}/{}".format(self.pilot_type, self.input_type)
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        weight_file = "{}/size{}_pilot{}_user{}.h5".format(weight_dir, self.size, self.pilot_len, self.user)
        gen_test_stage.save_weights(weight_file)


    def adjast_learning_rate(self):

        lr1 = 1e-2
        lr2 = 1e-3

        # if (self.NMSE <= -25):
        #     lr1 = 1e-4
        #     lr2 = 1e-3
        # if (self.NMSE <= -27):
        #     lr1 = 5e-5
        #     lr2 = 5e-4
        # if (self.NMSE <= -28):
        #     lr1 = 1e-5
        #     lr2 = 1e-5

        return lr1, lr2


    def create_weight(self):

        gen_weight = 1
        dis_weight = 0

        # if (self.NMSE <= -25):
        #     gen_weight = 0.98
        #     dis_weight = 0.02
        # if (self.NMSE <= -27):
        #     gen_weight = 0.97
        #     dis_weight = 0.03
        # if (self.NMSE <= -28):
        #     gen_weight = 0.95
        #     dis_weight = 0.05

        return gen_weight, dis_weight


