
import csv
import os
import pathlib

import math
import numpy as np

from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Activation, PReLU, ReLU
from keras.optimizers import Adam

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCH_COLUMN = 'Epoch'
NMSE_COLUMN = 'NMSE'

class GAN:
    def __init__(self, max_epoch, batch_size, sample, test, user, BS, pilot_type, pilot_len, size, input_type):
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.sample = sample
        self.test = test
        self.user = user
        self.BS = BS
        self.pilot_type = pilot_type
        self.pilot_len = pilot_len
        if input_type == 'ls':
            self.input_len = user
        if input_type == 'receive':
            self.input_len = pilot_len
        self.size = size
        self.input_type = input_type
        self.NMSE = 0
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
        input = Input(shape=(self.input_len*2,))

        x = Dense(units=self.input_len*4)(input)
        x = PReLU()(x)
        # x = ReLU()(x)
        # x = BentLinear()(x)
        # x = Activation('linear')(x)

        x = Dense(units=self.input_len*3)(x)
        x = PReLU()(x)
        # x = ReLU()(x)
        # x = BentLinear()(x)
        # x = Activation('linear')(x)

        output = Dense(units=self.user*2)(x)

        return Model(inputs=input, outputs=output, name='generator')


    def discriminator(self):
        input = Input(shape=(self.user*2,))

        x = Dense(units=64)(input)
        x = ReLU()(x)

        x = Dense(units=32)(x)
        x = ReLU()(x)

        output = Dense(units=2, activation='softmax')(x)

        return Model(inputs=input, outputs=output, name='discriminator')


    def create_model(self):
        input = Input(shape=(self.input_len*2,))
        true = Input(shape=(self.user*2,))

        gen = self.generator()
        gen_output = gen(input)

        dis = self.discriminator()
        dis_output = dis(gen_output)
        dis_true = dis(true)

        self.set_trainable(gen, True)
        self.set_trainable(dis, False)
        gen_train_stage = Model(inputs=[input, true], outputs=[gen_output, dis_output], name='gen_train_stage')
        gen_train_optimizer = Adam(beta_1=0.9)
        gen_train_stage.compile(optimizer=gen_train_optimizer,
                                loss={'generator': 'mean_absolute_error', 'discriminator': 'categorical_crossentropy'},
                                loss_weights={'generator': self.gen_weight, 'discriminator': self.dis_weight})

        self.set_trainable(gen, False)
        self.set_trainable(dis, True)
        dis_train_stage = Model(inputs=[input, true], outputs=[dis_output, dis_true], name='dis_train_stage')
        dis_train_optimizer = Adam(beta_1=0.9)
        dis_train_stage.compile(optimizer=dis_train_optimizer, loss='categorical_crossentropy',
                                metrics=['categorical_accuracy'])

        self.set_trainable(gen, True)
        self.set_trainable(dis, False)
        gen_test_stage = Model(inputs=input, outputs=gen_output, name='gen_test_stage')
        gen_test_optimizer = Adam(lr=0.0001, beta_1=0.9)
        gen_test_stage.compile(optimizer=gen_test_optimizer, loss='mean_absolute_error')

        return gen_train_stage, dis_train_stage, gen_test_stage


    def calculate_NMSE(self, epoch, input_test, answer_test, gen_test_stage):
        self.gen_weight, self.dis_weight = self.create_weight()
        tmp_true = np.zeros((self.user * self.BS * self.test), dtype=np.complex)
        tmp_deeplearning = np.zeros((self.user * self.BS * self.test), dtype=np.complex)

        for i in range(self.test):

            input_tmp = np.zeros((1, self.input_len*2), dtype=np.float)
            channel = np.zeros((1, self.user), dtype=np.complex)
            estimated_channel = np.zeros((1, self.user), dtype=np.complex)

            for j in range(self.input_len*2):
                input_tmp[0, j] = input_test[i, j]

            predict_test = gen_test_stage.predict(input_tmp, batch_size=1)

            for j in range(self.user):
                channel[0, j] = answer_test[i, j] + 1j * answer_test[i, j+self.user]
                estimated_channel[0, j] = predict_test[0, j] + 1j * predict_test[0, j+self.user]

            for k in range(self.user):
                tmp_true[self.user * i + k] = channel[0][k]
                tmp_deeplearning[self.user * i + k] = estimated_channel[0][k]

        self.NMSE = 10 * math.log10(np.sum(np.abs(tmp_true - tmp_deeplearning) * np.abs(tmp_true - tmp_deeplearning)) / np.sum(np.abs(tmp_true) * np.abs(tmp_true)))
        self.file_writing(epoch, self.NMSE)


    def file_writing(self, epoch, NMSE):

        NMSE_dir = "Data/epoch_data/{}/{}".format(self.pilot_type, self.input_type)
        if not os.path.exists(NMSE_dir):
            os.makedirs(NMSE_dir)

        NMSE_file = "{}/size{}_pilot{}_user{}.csv".format(NMSE_dir, self.size, self.pilot_len, self.user)
        if not os.path.exists(NMSE_file):
            pathlib.Path(NMSE_file).touch()

        NMSE_columns = [EPOCH_COLUMN, NMSE_COLUMN]

        if epoch == 0:
            with open(NMSE_file, 'w+') as f:
                writer = csv.DictWriter(f, fieldnames=NMSE_columns)
                writer.writeheader()

        with open(NMSE_file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=NMSE_columns)
            writer.writerow({
                EPOCH_COLUMN: epoch,
                NMSE_COLUMN: NMSE
            })

        print('--------------------------------')
        print('EPOCH: {}'.format(epoch))
        print('PILOT TYPE: {}'.format(self.pilot_type))
        print('INPUT TYPE: {}'.format(self.input_type))
        print('NMSE: {}'.format(self.NMSE))
        print('--------------------------------')

        return NMSE


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

        if (self.NMSE <= -25):
            lr1 = 1e-4
            lr2 = 1e-3
        if (self.NMSE <= -27):
            lr1 = 5e-5
            lr2 = 5e-4
        if (self.NMSE <= -28):
            lr1 = 1e-5
            lr2 = 1e-5

        return lr1, lr2


    def create_weight(self):

        gen_weight = 1
        dis_weight = 0

        if (self.NMSE <= -25):
            gen_weight = 0.98
            dis_weight = 0.02
        if (self.NMSE <= -27):
            gen_weight = 0.97
            dis_weight = 0.03
        if (self.NMSE <= -28):
            gen_weight = 0.95
            dis_weight = 0.05

        return gen_weight, dis_weight


