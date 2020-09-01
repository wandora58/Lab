
import math
import numpy as np
from itertools import product

from sympy.combinatorics.graycode import GrayCode


class Modulation:

    def __init__(self, M, code_symbol, data_symbol, bit, user, BS):
        """

        modulation class.

        Args:
            M : int = 4, 16, 64, 256
                  M-ary modulation number

            array_bit : 1D list [log2(M)]
                          Gray code list

                            ex) M = 4   : ['0', '1']
                                M = 16  : ['00', '01', '11', '10']
                                M = 64  : ['000', '001', '011', '010', '110', '111', '101', '100']
                                M = 256 : ['0000', '0001', '0011', '0010', '0110', '0111', '0101', '0100',
                                           '1100', '1101', '1111', '1110', '1010', '1011', '1001', '1000']


            array_power : 1D ndarray [log2(M)]
                            the power list at placement point

                            ex) M = 4   : [-0.7071068  0.7071068]
                                M = 16  : [-0.9486833 -0.3162278  0.3162278  0.9486833]
                                M = 64  : [-1.0801234 -0.7715167 -0.46291   -0.1543033  0.1543033  0.46291    0.7715167  1.0801234]
                                M = 256 : [-1.1504475 -0.9970545 -0.8436615 -0.6902685 -0.5368755 -0.3834825 -0.2300895 -0.0766965
                                            0.0766965  0.2300895  0.3834825  0.5368755  0.6902685  0.8436615  0.9970545  1.1504475]


            threshould_power : 1D ndarray [log2(M) - 1 ]
                                 the power threshold for demodulation

                                 ex) M = 4   : [0.]
                                     M = 16  : [-0.6324555  0.         0.6324555]
                                     M = 64  : [-0.9258201 -0.6172134 -0.3086067  0.         0.3086067  0.6172134 0.9258201]
                                     M = 256 : [-1.073751  -0.920358  -0.766965  -0.613572  -0.460179  -0.306786  -0.153393  0.
                                                 0.153393   0.306786   0.460179   0.613572   0.766965   0.920358  1.073751]

        """

        self.M = M
        self.symbol = code_symbol
        self.data_symbol = data_symbol
        self.bit = bit
        self.user = user
        self.BS = BS
        self.array_bit = list((GrayCode(int(bit/2)).generate_gray()))
        self.array_power, self.threshould_power = self.calculate_power(M, bit)


    def calculate_power(self, M, bit):
        """

        Calculate power at placement point

        """

        l1 = np.array([i ** 2 for i in range(1,int(M**(1/2)),2)])
        l2 = np.array([i ** 2 for i in range(1,int(M**(1/2)),2)])

        sum = 0
        for i in l1:
            for j in l2:
                sum += i+j

        norm_power = math.sqrt(M/(sum*4))

        l1 = np.array([i for i in range(1,int(M**(1/2)),2)])
        l2 = np.array([i for i in range(1,int(M**(1/2)),2)])

        l1 = norm_power * l1
        l2 = -1 * norm_power * l2
        array_power = np.concatenate([np.fliplr([l2])[0], l1], axis=0)

        t1 = np.array([i*norm_power for i in range(2,2**int(bit/2),2)])
        t2 = np.array([-1*i*norm_power for i in range(2,2**int(bit/2),2)])

        t1 = np.insert(t1,0,0)
        threshould_power = np.concatenate([np.fliplr([t2])[0], t1], axis=0)

        return np.round(array_power, decimals=7), np.round(threshould_power, decimals=7)


    def modulation(self, send_bit):
        """

        M-QAM modulation function.

        Args:
            send_bit : 2D ndarray [user, symbol * bit_rate * 1/code_rate]

        Returns:
            send_signal : 2D ndarray [user, symbol * 1/code_rate]
                            send signal after M-QAM modulation of send bit

                                us1  [[symbol11, symbol12, symbol13]
                                us2   [symbol21, symbol22, symbol23]
                                us3   [symbol31, symbol32, symbol33]
                                      [  ...  ,    ...   ,   ...   ]]

        """
        send_signal = np.zeros((self.user, self.symbol), dtype=np.complex)

        for s in range(self.user):
            for i in range(self.symbol):
                bit_I = ''
                bit_Q = ''
                for j in range(int(self.bit/2)):
                    bit_I += str(send_bit[s][self.bit*i+j])
                    bit_Q += str(send_bit[s][self.bit*i+j + int(self.bit/2)])

                idx_I = self.array_bit.index(bit_I)
                idx_Q = self.array_bit.index(bit_Q)

                send_signal[s,i] = self.array_power[idx_I] + 1j*self.array_power[idx_Q]

        return np.round(send_signal, decimals=7)



    def demodulation(self, receive_signal, demod_type, sigma):
        """

        M-QAM demodulation fuction.

        Args:
            receive_signal (np.array): receive signal

        Returns:
            receive_bit (np.array): receive bit after QPSK demodulation of receive signal

        """

        if demod_type == 'hard':
            receive_bit = [[] for _ in range(self.BS)]
            for r in range(self.BS):
                for i in range(self.symbol):

                    tmp = receive_signal[r][i]

                    tmp_I = np.real(tmp)
                    tmp_Q = np.imag(tmp)

                    idx_I = 0
                    idx_Q = 0
                    for j in range(len(self.threshould_power)):
                        if tmp_I >= self.threshould_power[j]:
                            idx_I = j+1

                        if tmp_Q >= self.threshould_power[j]:
                            idx_Q = j+1

                    for j in range(self.bit):
                        if j < int(self.bit/2):
                            receive_bit[r].append(int(self.array_bit[idx_I][j]))

                        else:
                            receive_bit[r].append(int(self.array_bit[idx_Q][j-int(self.bit/2)]))

            return np.array(receive_bit)

        if demod_type == 'soft':
            receive_bit = np.zeros((self.BS, self.symbol * self.bit))

            constellation = []
            for i in self.array_power:
                for j in self.array_power:
                    constellation.append(i + 1j*j)

            array_bit = []
            for i in self.array_bit:
                for j in self.array_bit:
                    array_bit.append(i+j)

            for r in range(self.BS):
                for i in range(self.symbol):
                    current_symbol = receive_signal[r, i]
                    for bit_index in range(self.bit):
                        llr_num = 0
                        llr_den = 0
                        for bit_value, symbol in zip(array_bit, constellation):

                            if bit_value[bit_index*-1 - 1] == '1':
                                llr_num += np.exp((-abs(current_symbol - symbol) ** 2) / (sigma*sigma))
                            else:
                                llr_den += np.exp((-abs(current_symbol - symbol) ** 2) / (sigma*sigma))
                                
                        receive_bit[r][i * self.bit + self.bit - 1 - bit_index] = np.log(llr_num / llr_den)

            return receive_bit

