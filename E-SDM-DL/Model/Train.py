
import csv
import os
import pathlib
import math
import numpy as np

from Model.Channel import Channel
from Model.Estimate import Estimate
from Model.Sequence import DFTSequence
from Model.Frame import Frame
from Model.utils import Noise


class Train:
    def __init__(self, flame=10500, symbol=128, user=16, BS=1, CP=1, path=1, Ps=1.0, EbN0=20, pilot_len=16, size=16, type=None):
        self.flame = flame
        self.symbol = symbol
        self.user = user
        self.BS = BS
        self.CP = CP
        self.path = path
        self.Ps = Ps
        self.EbN0 = EbN0
        self.sigma = math.sqrt(0.5 * math.pow(10.0, (-1) * EbN0 * 0.1))
        self.pilot_len = pilot_len
        self.size = size
        self.type = type
        if self.type == 'DFT':
            self.pilot_signal = DFTSequence(user, path, pilot_len, size).create_pilot()
        if self.type == 'frame':
            self.pilot_signal = Frame(user, pilot_len).create_frame_matrix()


    def create_train(self):
        for sample in range(self.flame):
            self.channel = Channel(self.symbol, self.user, self.BS, 1, self.path, self.CP, self.Ps).create_rayleigh_channel()
            estimated_channel, receive_pilot_signal = Estimate(self.pilot_signal, self.channel, self.user, self.BS, self.path, self.sigma, 1).least_square_estimate()
            self.file_writing(sample, estimated_channel.reshape(1,self.user), receive_pilot_signal.reshape(1,self.pilot_len), self.channel.reshape(1,self.user))


    def file_writing(self, sample, estimated_channel, receive_pilot_signal, channel):

        dir = "Data/train_data/{}".format(self.type)
        if not os.path.exists(dir):
            os.makedirs(dir)

        ls_dir = "{}/ls".format(dir)
        if not os.path.exists(ls_dir):
            os.makedirs(ls_dir)

        ls_file = "{}/size{}_pilot{}_user{}.csv".format(ls_dir, self.size, self.pilot_len, self.user)
        if not os.path.exists(ls_file):
            pathlib.Path(ls_file).touch()

        if sample == 0:
            with open(ls_file, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(np.concatenate([np.real(estimated_channel), np.imag(estimated_channel)], 1))

        else:
            with open(ls_file, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(np.concatenate([np.real(estimated_channel), np.imag(estimated_channel)], 1))



        receive_dir = "{}/receive".format(dir)
        if not os.path.exists(receive_dir):
            os.makedirs(receive_dir)

        receive_file = "{}/size{}_pilot{}_user{}.csv".format(receive_dir, self.size, self.pilot_len, self.user)
        if not os.path.exists(receive_file):
            pathlib.Path(receive_file).touch()

        if sample == 0:
            with open(receive_file, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(np.concatenate([np.real(receive_pilot_signal), np.imag(receive_pilot_signal)], 1))

        else:
            with open(receive_file, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(np.concatenate([np.real(receive_pilot_signal), np.imag(receive_pilot_signal)], 1))



        true_dir = "{}/true".format(dir)
        if not os.path.exists(true_dir):
            os.makedirs(true_dir)

        true_file = "{}/size{}_pilot{}_user{}.csv".format(true_dir, self.size, self.pilot_len, self.user)
        if not os.path.exists(true_file):
            pathlib.Path(true_file).touch()

        if sample == 0:
            with open(true_file, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(np.concatenate([np.real(channel), np.imag(channel)], 1))

        else:
            with open(true_file, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(np.concatenate([np.real(channel), np.imag(channel)], 1))



