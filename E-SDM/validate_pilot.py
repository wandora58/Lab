import math
import numpy as np
from Model.Sequence import ZadoffSequence, DFTSequence
from Model.Frame import Frame
from Model.Channel import Channel
from Model.Estimate import Estimate

user = 19
BS = 2
path = 1
pilot_len = 9
size = 19
symbol = 128
Nc = 128
CP = 1
Ps = 1.0
EbN0 = 30
sigma = math.sqrt( 0.5 * math.pow(10.0 , (-1) * EbN0 * 0.1))

channel = Channel(symbol, user, BS, Nc, path, CP, Ps)
true_channel = channel.create_rayleigh_channel()

dft_pilot = DFTSequence(user, path, pilot_len, size).create_pilot()
frame_pilot = Frame(user, pilot_len).create_frame_matrix()

for r in range(BS):
    tmp_channel = np.zeros((user*path, 1),dtype=np.complex)

    for s in range(user):
        for p in range(path):
            tmp_channel[s*path + p, 0] = true_channel[p, user*r+s]

    tmp_receive = np.dot(frame_pilot, tmp_channel)

    A_ = np.linalg.pinv(frame_pilot)
    y_ = np.dot(A_, tmp_receive)

    A = frame_pilot
    Ah = np.conjugate(A.T)
    y_1 = Ah @ np.linalg.inv(A @ Ah) @ tmp_receive
    # y_1 = np.dot(np.linalg.inv(A), tmp_receive)

    for i in range(user):
        print(y_[i,0], y_1[i,0], true_channel[0,user*r+i], tmp_channel[i,0])



A = np.conjugate(dft_pilot.T) @ dft_pilot
l = []
for i in range(user):
    for j in range(user):
        if round(A[i][j], 8) not in l:
            l.append(round(A[i][j], 8))

dft_channel = Estimate(dft_pilot, true_channel, user, BS, path, sigma, Nc).least_square_estimate()


B = np.conjugate(frame_pilot.T) @ frame_pilot
l1 = []
for i in range(user):
    for j in range(user):
        if round(B[i][j], 8) not in l1:
            l1.append(round(B[i][j], 8))

frame_channel = Estimate(frame_pilot, true_channel, user, BS, path, sigma, Nc).least_square_estimate()
