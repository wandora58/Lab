
import numpy as np
import math

from scipy import linalg
from tqdm import tqdm

from Model.Channel import Channel
from Model.Selection import Selection
from Model.utils import MMSE_channel_capacity


user = 6 # ユーザー数
user_antenna = 1 # ユーザのアンテナ数

BS = 1 # 基地局数
BS_antenna = 4 # 基地局のアンテナ数

select_user = math.floor(BS_antenna/user_antenna) # 選択されたユーザ数

Nc = 72 # 搬送波数

CP = 1 # CP数
path = 1 # パス数
Ps = 1.0 # 総送信電力

SNRdB = 10
SNR = 10 ** (SNRdB / 10)

symbol = 72

frame = 100

channel = Channel(user, user_antenna, BS_antenna, Nc, path, CP, Ps, symbol)
selection = Selection(user, user_antenna, BS, BS_antenna, select_user)

CDUS = []
RAND = []
ALL = []
for SNRdB in tqdm(range(0,31)):
    SNR = 10 ** (SNRdB / 10)
    cdus = 0
    rand = 0
    all = 0

    for _ in range(frame):

        true_channel = channel.create_rayleigh_channel()

        cdus_channel = selection.CDUS(true_channel)
        cdus += MMSE_channel_capacity(cdus_channel, select_user, user_antenna, BS_antenna, SNR)

        rand_channel = selection.RAND(true_channel)
        rand += MMSE_channel_capacity(rand_channel, select_user, user_antenna, BS_antenna, SNR)

        all_channel, tmp = selection.ALL(true_channel, SNR)
        all += tmp

    CDUS.append(cdus/frame)
    RAND.append(rand/frame)
    ALL.append(all/frame)

for i in range(0,31):
    print(CDUS[i], RAND[i], ALL[i])