
import csv
import os
import pathlib

import numpy as np
import math
import random

COMMON_COLUMN = 'SNR'
BER_COLUMN = 'BER'
NMSE_COLUMN = 'NMSE'

class Weight:
    def __init__(self):
        pass

    def create_weight(self, path):
        w = 1
        ww = 1

        for i in range(path-1):
            w *= pow(10,-0.1)
            ww += w

        weight = math.sqrt(ww)

        return weight


class Boxmuller:
    def __init__(self):
        pass

    def create_normalized_random(self, k):
        a = random.random()
        b = random.random()

        x = k * math.sqrt(-2 * math.log(a)) * math.sin(2 * math.pi * b)
        y = k * math.sqrt(-2 * math.log(a)) * math.cos(2 * math.pi * b)

        z = x + y * 1j

        return z


class Noise(Boxmuller):
    def __init__(self):
        pass

    def create_noise(self, symbol, sigma, BS):
        """

        Create White Gaussian noise

        Args:
            symbol (int): the number of symbol
            sigma (float): standard deviations
            BS (int): the number of symbol

        Returns:
            noise (np.array): Rayleigh fading channel with 1 dB attenuation by path

                                            BS1           BS2
                                         u1  u2 ... /  u1 us2 ... /
                               path1 [[ h11 h21 h31   h12 h22 h32 ],
                               path2  [ h11 h21 h31   h12 h22 h32 ],                          ]
                               path3  [ h11 h21 h31   h12 h22 h32 ],
                                                   ...            ]]

                                  hij : Channel from i-th user to j-th BS
                                        1db attenuation for each additional path


        """
        noise = np.zeros((symbol, BS), dtype=np.complex)

        for s in range(BS):
            for i in range(symbol):
                noise[i, s] = super().create_normalized_random(sigma)

        return noise


class CP:
    def __init__(self, CP, symbol):
        self.CP = CP
        self.symbol = symbol

    def add_CP(self):
        O = np.zeros((self.CP, self.symbol - self.CP))
        Icp = np.eye(self.CP, k=0)
        I = np.eye(self.symbol, k=0)

        return np.vstack([np.hstack([O, Icp]), I])


    def remove_CP(self):
        O = np.zeros((self.symbol, self.CP))
        I = np.eye(self.symbol, k=0)

        return np.hstack([O, I])


class Result:
    def __init__(self, flame, user, BS, symbol, Nc):
        self.flame = flame
        self.user = user
        self.BS = BS
        self.symbol = symbol
        self.Nc = Nc

        self.tmp_true = np.zeros((Nc*user*BS),dtype=np.complex)
        self.tmp_estimated = np.zeros((Nc*user*BS),dtype=np.complex)

        self.h_true = np.zeros((flame*Nc*user*BS),dtype=np.complex)
        self.h_estimated = np.zeros((flame*Nc*user*BS),dtype=np.complex)

        self.rrs = 0
        self.rrn = 0
        self.BER = 0
        self.NMSE = 0


    def calculate(self, count, send_symbol, receive_symbol, true_channel, estimated_channel, send_signal, noise, EbN0, type):
        self.count = count
        self.send_symbol = send_symbol
        self.receive_symbol = receive_symbol
        self.true_channel = true_channel
        self.estimated_channel = estimated_channel
        self.send_signal = send_signal
        self.noise = noise
        self.EbN0 = EbN0
        self.type = type

        # SNR
        self.rrs += np.sum(np.abs(self.send_signal) * (np.abs(self.send_signal)))
        self.rrn += np.sum(np.abs(self.noise) * np.abs(self.noise))

        # BER
        for r in range(self.user):
            for i in range(self.symbol):

                if self.send_symbol[i][r] != self.receive_symbol[i][r]:
                    self.BER += 1

        # NMSE
        for j in range(self.user*self.BS):
            for k in range(self.Nc):

                self.tmp_true[self.Nc*j+k] = self.true_channel[k][j]
                self.tmp_estimated[self.Nc*j+k] = self.estimated_channel[k][j]

        for i in range(self.Nc*self.user*self.BS):

            self.h_true[self.count*self.Nc*self.user*self.BS+i] = self.tmp_true[i]
            self.h_estimated[self.count*self.Nc*self.user*self.BS+i] = self.tmp_estimated[i]

        if self.count+1 == self.flame:
            self.file_writing()


    def file_writing(self):

        BER_columns = [COMMON_COLUMN, BER_COLUMN]
        BER_dir = "BER/{}".format(self.type)
        BER_file = "{}/{}_{}.csv".format(BER_dir, self.user, self.BS)

        NMSE_columns = [COMMON_COLUMN, NMSE_COLUMN]
        NMSE_dir = "NMSE/{}".format(self.type)
        NMSE_file = "{}/{}_{}.csv".format(NMSE_dir, self.user, self.BS)

        if not os.path.exists(BER_dir):
            os.makedirs(BER_dir)

        if not os.path.exists(BER_file):
            pathlib.Path(BER_file).touch()

        if not os.path.exists(NMSE_dir):
            os.makedirs(NMSE_dir)

        if not os.path.exists(NMSE_file):
            pathlib.Path(NMSE_file).touch()

        if self.EbN0 == 0:
            with open(BER_file, 'w+') as f:
                writer = csv.DictWriter(f, fieldnames=BER_columns)
                writer.writeheader()

            with open(NMSE_file, 'w+') as f:
                writer = csv.DictWriter(f, fieldnames=NMSE_columns)
                writer.writeheader()


        self.SNR = 10 * math.log10(self.rrs/self.rrn)
        self.BER /= self.flame * self.symbol * self.user
        if self.true_channel[0,0] == self.estimated_channel[0,0]:
            self.NMSE = -float('inf')
        else:
            self.NMSE = 10 * math.log10(np.sum(np.abs(self.h_true-self.h_estimated)*np.abs(self.h_true-self.h_estimated)) / np.sum(np.abs(self.h_estimated)*np.abs(self.h_estimated)))

        with open(BER_file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=BER_columns)
            writer.writerow({
                COMMON_COLUMN: self.SNR,
                BER_COLUMN: self.BER
            })

        with open(NMSE_file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=NMSE_columns)
            writer.writerow({
                COMMON_COLUMN: self.SNR,
                NMSE_COLUMN: self.NMSE
            })

        print('--------------------------------')
        print('TYPE: {}'.format(self.type))
        print('SNR: {}'.format(self.SNR))
        print('BER: {}'.format(self.BER))
        print('NMSE: {}'.format(self.NMSE))
        print('--------------------------------')


class Correlation:
    def __init__(self, pilot_signal):
        self.pilot_signal = pilot_signal

    def create_cor_matirix(self):
        pilot_len = np.shape(self.pilot_signal)[0]
        user = np.shape(self.pilot_signal)[1]

        cor_matrix = np.zeros((pilot_len, pilot_len), dtype=np.complex)
        cor_tmp1 = np.zeros((pilot_len, 1), dtype=np.complex)
        cor_tmp2 = np.zeros((pilot_len, 1), dtype=np.complex)

        for i in range(pilot_len):

            for j in range(pilot_len):
                cor_tmp1[j,0] = pilot_signal[i,j]

            for j in range(pilot_len):

                for k in range(pilot_len):
                    cor_tmp2[k,0] = pilot_signal[j,k]

                cor_matrix[i,j] = np.sum(cor_tmp1 * np.conjugate(cor_tmp2)) / user

        return cor_matrix


class Load:
    def __init__(self, sample, test, size, pilot_len, user, pilot_type, input_type):
        self.sample = sample
        self.test = test
        self.size = size
        self.pilot_len = pilot_len
        self.user = user
        self.pilot_type = pilot_type
        self.input_type = input_type

        self.input = "Data/train_data/{}/{}/size{}_pilot{}_user{}.csv".format(pilot_type, input_type, size, pilot_len, user)
        self.true = "Data/train_data/{}/true/size{}_pilot{}_user{}.csv".format(pilot_type, size, pilot_len, user)


    def load_data(self):

        input = []
        answer = []

        with open(self.input, 'r') as f:
            for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                input.append(row)

        with open(self.true, 'r') as f:
            for row in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC):
                answer.append(row)

        input_data = input[:self.sample][:]
        input_test = input[self.sample:][:]

        answer_data = answer[:self.sample][:]
        answer_test = answer[self.sample:][:]

        return np.array(input_data), np.array(input_test), np.array(answer_data), np.array(answer_test)
