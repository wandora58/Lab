
import csv
import os
import pathlib

import itertools
import numpy as np
import math
import random
import matplotlib

from scipy.interpolate import interp1d

matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

        z = x + y*1j

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
        noise = np.zeros((BS, symbol), dtype=np.complex)

        for r in range(BS):
            for i in range(symbol):
                noise[r, i] = (super().create_normalized_random(sigma))

        noise /= np.sqrt(BS)

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


class Bitrate:
    def __init__(self, M):
        self.M = M

    def count_bit_rate(self):
        """

        Calculates the number of bits per symbol from the M-ary modulation number M

        Returns:
            bit_rate (int): the number of bits per symbol

        """

        bit_rate = 0
        while(True):
            if 2 ** bit_rate == self.M:
                break
            bit_rate += 1

        return bit_rate


class Serial2Parallel:
    def __init__(self, data_stream, bit_rate, code_symbol):
        self.data_stream = data_stream
        self.bit_rate = bit_rate
        self.symbol = code_symbol

    def create_parallel(self):
        """
        Serial signal to Parallel signal

        Args:
            data_stream : 1D ndarray [data_symbol * total_bit]

        Returns:
            send_bit : 2D ndarray

        """
        send_bit = []
        tmp = 0
        for i in range(len(self.bit_rate)):
            if i == 0:
                send_bit.append(self.data_stream[: self.symbol*self.bit_rate[i]])
            else:
                send_bit.append(self.data_stream[tmp : tmp + self.symbol*self.bit_rate[i]])

            tmp += self.symbol*self.bit_rate[i]

        return send_bit


class Parallel2Serial:
    def __init__(self, receive_symbol, bit_rate, symbol):
        self.receive_symbol = receive_symbol
        self.bit_rate = bit_rate
        self.symbol = symbol

    def create_serial(self):
        return list(itertools.chain.from_iterable(self.receive_symbol))




class Result:
    def __init__(self, flame, user, BS, symbol, Nc):
        self.flame = flame
        self.user = user
        self.BS = BS
        self.symbol = symbol
        self.Nc = Nc
        self.rrs = 0
        self.rrn = 0
        self.BER = 0


    def calculate(self, count, send_data, receive_data, send_signal, noise, EbN0, total_bit):
        self.count = count
        self.send_data = send_data
        self.receive_data = receive_data
        self.send_signal = send_signal
        self.noise = noise
        self.EbN0 = EbN0
        self.total_bit = total_bit

        # SNR
        self.rrs += np.sum(np.abs(self.send_signal) * np.abs(self.send_signal))
        self.rrn += np.sum(np.abs(self.noise) * np.abs(self.noise))

        # BER
        for i in range(self.symbol * self.total_bit):
            if self.send_data[i] != self.receive_data[i]:
                self.BER += 1

        if self.count+1 == self.flame:
            self.file_writing()

            if self.EbN0 == 30:
                self.illustrate()


    def file_writing(self):

        BER_columns = [COMMON_COLUMN, BER_COLUMN]
        BER_file = "BER/user{}_BS{}_bit{}.csv".format(self.user, self.BS, self.total_bit)

        if not os.path.exists(BER_file):
            pathlib.Path(BER_file).touch()

        if self.EbN0 == 0:
            with open(BER_file, 'w+') as f:
                writer = csv.DictWriter(f, fieldnames=BER_columns)
                writer.writeheader()

        self.SNR = 10 * math.log10(self.rrs/self.rrn)
        self.BER /= self.flame * self.symbol * self.total_bit

        with open(BER_file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=BER_columns)
            writer.writerow({
                COMMON_COLUMN: self.SNR,
                BER_COLUMN: self.BER
            })

        print('--------------------------------')
        print('BIT: {}'.format(self.total_bit))
        print('SNR: {}'.format(self.SNR))
        print('BER: {}'.format(self.BER))
        print('--------------------------------')


    def illustrate(self):

        BER = np.zeros(31,dtype=np.float)
        x = np.zeros(31,dtype=np.float)

        BER_file = "BER/user{}_BS{}_bit{}.csv".format(self.user, self.BS, self.total_bit)

        with open(BER_file, 'r') as f:
            h = next(csv.reader(f))
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            i = 0
            for row in reader:
                x[i] = row[0]
                BER[i] = row[1]
                i += 1

        def spline_interp(in_x, in_y):
            out_x = np.linspace(np.min(in_x), np.max(in_x), 22)# もとのxの個数より多いxを用意
            func_spline = interp1d(in_x, in_y, kind='cubic') # cubicは3次のスプライン曲線
            out_y = func_spline(out_x) # func_splineはscipyオリジナルの型

            return out_x, out_y

        x, BER = spline_interp(x, BER)

        #--------------BER---------------------------------------

        #figure　
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        #plot
        ax.plot(x, BER, linestyle='-', color='k', label='Adaptive modulation')

        #y軸の範囲設定
        ax.set_xlim([0,25])
        ax.set_ylim([0.0001,0.5])

        #y軸を片対数
        ax.set_yscale('log')

        #ひげ消す
        plt.gca().xaxis.set_tick_params(direction='in')
        plt.gca().yaxis.set_tick_params(direction='in')

        #x軸間隔
        plt.xticks([0,5,10,15,20,25,30])

        #x軸,y軸のラベル付け
        ax.set_xlabel('Average SNR [dB]', fontsize=12)
        ax.set_ylabel('Average BER', fontsize=12)

        #グリッド表示
        plt.grid(which="both")

        #凡例とタイトル
        ax.legend(loc='best',prop={'size':12})

        #保存　
        plt.savefig('Image/BER/user{}_BS{}_bit{}.pdf'.format(self.user, self.BS, self.total_bit))


def create_cor_matirix(pilot_signal):
    pilot_len = np.shape(pilot_signal)[0]
    user = np.shape(pilot_signal)[1]

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


def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer or an array-like of positive integers to NumPy array of the specified size containing
    bits (0 and 1).
    Parameters
    ----------
    in_number : int or array-like of int
        Positive integer to be converted to a bit array.
    bit_width : int
        Size of the output bit array.
    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).
    """

    if isinstance(in_number, (np.integer, int)):
        return decimal2bitarray(in_number, bit_width)
    result = np.zeros(bit_width * len(in_number), np.int8)
    for pox, number in enumerate(in_number):
        result[pox * bit_width:(pox + 1) * bit_width] = decimal2bitarray(number, bit_width)
    return result


def decimal2bitarray(number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing bits (0 and 1). This version is slightly
    quicker that dec2bitarray but only work for one integer.
    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.
    bit_width : int
        Size of the output bit array.
    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).
    """
    result = np.zeros(bit_width, np.int8)
    i = 1
    pox = 0
    while i <= number:
        if i & number:
            result[bit_width - pox - 1] = 1
        i <<= 1
        pox += 1
    return result


def bitarray2dec(in_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.
    Parameters
    ----------
    in_bitarray : 1D ndarray of ints
        Input NumPy array of bits.
    Returns
    -------
    number : int
        Integer representation of input bit array.
    """

    number = 0

    for i in range(len(in_bitarray)):
        number = number + in_bitarray[i] * pow(2, len(in_bitarray) - 1 - i)

    return number


def hamming_dist(in_bitarray_1, in_bitarray_2):
    """
    Computes the Hamming distance between two NumPy arrays of bits (0 and 1).
    Parameters
    ----------
    in_bit_array_1 : 1D ndarray of ints
        NumPy array of bits.
    in_bit_array_2 : 1D ndarray of ints
        NumPy array of bits.
    Returns
    -------
    distance : int
        Hamming distance between input bit arrays.
    """

    distance = np.bitwise_xor(in_bitarray_1, in_bitarray_2).sum()

    return distance


def euclid_dist(in_array1, in_array2):
    """
    Computes the squared euclidean distance between two NumPy arrays
    Parameters
    ----------
    in_array1 : 1D ndarray of floats
        NumPy array of real values.
    in_array2 : 1D ndarray of floats
        NumPy array of real values.
    Returns
    -------
    distance : float
        Squared Euclidean distance between two input arrays.
    """
    distance = ((in_array1 - in_array2) * (in_array1 - in_array2)).sum()

    return distance


def signal_power(signal):
    """
    Compute the power of a discrete time signal.
    Parameters
    ----------
    signal : 1D ndarray
             Input signal.
    Returns
    -------
    P : float
        Power of the input signal.
    """

    @np.vectorize
    def square_abs(s):
        return abs(s) ** 2

    P = np.mean(square_abs(signal))
    return P



def cumulative_probability(channel, user, BS):
    count = 100000
    matrix = np.zeros((count,user), dtype=np.float)

    for cnt in range(count):
        h = channel.create_rayleigh_channel()
        H = np.zeros((BS, user), dtype=np.complex)

        for r in range(BS):
            for s in range(user):
                H[r][s] = h[0][BS*r + s]

        gram_H = np.conjugate(H.T) @ H
        U, A ,Uh = svd(gram_H)

        for s in range(user):
            matrix[cnt][s] = 10 * math.log10(A[s])

    matrix = np.sort(matrix,axis=0)

    x = range(-40,20)
    cumulative_matrix = np.zeros((user, len(x)),dtype=np.double)

    for i in range(len(x)) :
        for s in range(user):
            tmp = np.where(matrix[:,s] <= x[i], 1, matrix[:,s])
            cumulative_matrix[s,i] = sum(tmp == 1) / count

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(user):
        ax.plot(x, cumulative_matrix[i], linestyle='-', color='k')

    plt.gca().xaxis.set_tick_params(direction='in')
    plt.gca().yaxis.set_tick_params(direction='in')

    ax.set_xlabel('eigenvalue [dB]', fontsize=12)
    ax.set_ylabel('Cumulative probability', fontsize=12)

    plt.grid(which="both")

    plt.savefig('Image/cdf/user{}_BS{}.pdf'.format(user, BS))

