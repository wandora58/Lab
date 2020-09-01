
import math
import numpy as np

from sympy.combinatorics.graycode import GrayCode


class Modulation:

    def __init__(self, user, BS):
        self.user = user
        self.BS = BS


    def calculate_power(self, M, bit):
        """

        Calculate power at placement point

        Returns:
            array_power (1d np.array): the power at placement point

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


    def modulation(self, send_bit, num_stream, bit_rate, code_symbol):
        """

        M-QAM modulation function.

        Args:
            send_bit (np.array): send bit

        Returns:
            send_signal (np.array): send signal after M-QAM modulation of send bit

                                us1  [[symbol11, symbol12, symbol13]
                                us2   [symbol21, symbol22, symbol23]
                                us3   [symbol31, symbol32, symbol33]
                                      [  ...  ,    ...   ,   ...   ]]

        """
        self.num_stream = num_stream
        self.bit_rate = bit_rate
        self.code_symbol = code_symbol

        send_signal = np.zeros((self.num_stream, self.code_symbol), dtype=np.complex)
        for s in range(self.num_stream):
            bit = self.bit_rate[s]
            M = 2 ** bit
            array_bit = list((GrayCode(int(bit/2)).generate_gray()))
            array_power, threshould_power = self.calculate_power(M, bit)

            for i in range(self.code_symbol):
                bit_I = ''
                bit_Q = ''
                for j in range(int(bit/2)):
                    bit_I += str(send_bit[s][bit*i+j])
                    bit_Q += str(send_bit[s][bit*i+j + int(bit/2)])

                idx_I = array_bit.index(bit_I)
                idx_Q = array_bit.index(bit_Q)

                send_signal[s,i] = array_power[idx_I] + 1j*array_power[idx_Q]

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
            receive_bit = [[] for _ in range(self.num_stream)]
            idx = 0
            for bit in self.bit_rate:
                M = 2 ** bit
                array_bit = list((GrayCode(int(bit/2)).generate_gray()))
                array_power, threshould_power = self.calculate_power(M, bit)

                for i in range(self.code_symbol):

                    tmp = receive_signal[idx][i]

                    tmp_I = np.real(tmp)
                    tmp_Q = np.imag(tmp)

                    idx_I = 0
                    idx_Q = 0
                    for j in range(len(threshould_power)):
                        if tmp_I >= threshould_power[j]:
                            idx_I = j+1

                        if tmp_Q >= threshould_power[j]:
                            idx_Q = j+1

                    for j in range(bit):
                        if j < int(bit/2):
                            receive_bit[idx].append(int(array_bit[idx_I][j]))

                        else:
                            receive_bit[idx].append(int(array_bit[idx_Q][j-int(bit/2)]))
                idx += 1


        if demod_type == 'soft':
            receive_bit = []
            for bit in self.bit_rate:
                receive_bit.append(np.zeros(self.code_symbol * bit, dtype=np.float64))

            for s in range(self.num_stream):
                bit = self.bit_rate[s]
                M = 2 ** bit
                array_bit = list((GrayCode(int(bit/2)).generate_gray()))
                array_power, threshould_power = self.calculate_power(M, bit)

                constellation = []
                for i in array_power:
                    for j in array_power:
                        constellation.append(i + 1j*j)

                array_pattern = []
                for i in array_bit:
                    for j in array_bit:
                        array_pattern.append(i + j)

                for i in range(self.code_symbol):
                    current_symbol = receive_signal[s, i]
                    for bit_index in range(bit):
                        llr_num = 0
                        llr_den = 0
                        for bit_value, symbol in zip(array_pattern, constellation):

                            if bit_value[bit_index*-1 - 1] == '1':
                                llr_num += np.exp((-abs(current_symbol - symbol) ** 2) / (sigma*sigma))
                            else:
                                llr_den += np.exp((-abs(current_symbol - symbol) ** 2) / (sigma*sigma))

                        receive_bit[s][i * bit + bit - 1 - bit_index] = np.log(llr_num / llr_den)

        return receive_bit
