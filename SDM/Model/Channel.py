
import numpy as np
import math

from Model import utils


class Channel:
    def __init__(self, code_symbol, user, BS, Nc, path, CP, Ps, channel_type):
        """

        Mimo channel class.

        Args:
            channel_type : str = 'AWGN', 'Rayleigh'

        """
        self.symbol = code_symbol
        self.user = user
        self.BS = BS
        self.Nc = Nc
        self.path = path
        self.CP = CP
        self.Ps = Ps
        self.channel_type = channel_type
        self.weight = utils.Weight().create_weight(path)
        self.channel = np.zeros((Nc, user * BS), dtype=np.complex)


    def create_channel(self):
        """

        Create channel

        Returns:
            channel : 2D ndarray [Nc, user * BS]

                                     BS1           BS2
                                 u1  u2 ... /  u1 us2 ... /
                        path1 [[ h11 h21 h31   h12 h22 h32 ],
                        path2  [ h11 h21 h31   h12 h22 h32 ],
                        path3  [ h11 h21 h31   h12 h22 h32 ],
                                         ...               ]]

                         hij : Channel from i-th user to j-th BS
                               if channel_type = 'AWGN'
                                  hij = 1

                               if channel_type = 'Rayleigh'
                                  hij = Boxmuller().create_normalized_random
                                        1db attenuation for each additional path

        """

        if self.channel_type == 'AWGN':
            for i in range(self.user * self.BS):
                for path in range(self.path):
                    self.channel[path, i] = 1


        elif self.channel_type == 'Rayleigh':
            for i in range(self.user * self.BS):
                Ps = self.Ps
                for path in range(self.path):
                    self.channel[path, i] = utils.Boxmuller().create_normalized_random(Ps / self.weight * math.sqrt(0.5))
                    # Ps *= math.sqrt(math.pow(10, -0.1))  # 1dB減衰

        return self.channel


    def channel_multiplication(self, send_signal):
        """

        Multiplies send signal by mimo channel

        Args:
            send_signal : 2D ndarray [user, symbol * 1/code_rate]

                              us1  [[symbol11, symbol12, symbol13]
                              us2   [symbol21, symbol22, symbol23]
                              us3   [symbol31, symbol32, symbol33]
                                    [  ...  ,    ...   ,   ...   ]]

                                     symbolij : j-th symbol of i-th user


        Returns:
            receive_signal : 2D ndarray [BS, symbol * 1/code_rate]
                               Received signal after multiplying send signal by channel

                               if path = 1:                    BS1           BS2
                                                            u1  u2 ... /  u1 us2 ... /
                                      channel  =  path1 [[ h11 h21 h31   h12 h22 h32 ],
                                                  path2  [   0   0   0     0   0   0 ],
                                                                   ...               ]]


                                            H  =  2D ndarray [BS, user]

                                                                user
                                                      BS1 [[h11 h21 h31 ..],
                                                      BS2  [h12 h22 h32 ..],
                                                           [      ...     ]]


                               receive_signal  =  2D ndarray [BS, symbol * 1/code_rate]

                                                                    H        *   send_signal
                                                               [BS, user]        [user, symbol * 1/code_rate]

                                                    BS1 [[h11 h21 h31 ..],   *   us1 [[symbol11, symbol12, symbol13]
                                                    BS2  [h12 h22 h32 ..],       us2  [symbol21, symbol22, symbol23]
                                                         [      ...     ]]            [            ...             ]]

                                                                             =

                                                            BS1 [[symbol11, symbol12, symbol13]
                                                            BS2  [symbol21, symbol22, symbol23]
                                                                 [            ...             ]]


        """
        self.send_signal = send_signal
        self.receive_signal = np.zeros((self.BS, self.symbol), dtype=np.complex)

        if self.path == 1:

            H = np.zeros((self.BS, self.user), dtype=np.complex)

            for r in range(self.BS):
                for s in range(self.user):
                    H[r][s] = self.channel[0][self.BS*r + s]

            self.receive_signal = np.dot(H, self.send_signal)


        elif self.path >= 1:
            h = np.zeros(self.path, dtype=np.complex)
            Hb = np.zeros((self.symbol + self.CP, self.symbol + self.CP), dtype=np.complex)
            H = np.zeros((self.symbol, self.symbol), dtype=np.complex)
            CP = utils.CP(self.CP, self.symbol)

            for r in range(self.BS):

                send_tmp = np.zeros((self.symbol, 1), dtype=np.complex)
                receive_tmp = np.zeros((self.symbol, 1), dtype=np.complex)

                for s in range(self.user):

                    for p in range(self.path):
                        h[p] = self.channel[p][r * BS + s]

                    k = 0
                    for count in range(self.path):
                        for i in range(self.symbol + self.CP):
                            for j in range(self.symbol + self.CP):

                                if (j + k == i):
                                    Hb[i, j] = h[k]  # Add CP

                        k += 1

                    H = np.dot(np.dot(CP.remove_CP(), Hb), CP.add_CP())  # Remove CP

                    for i in range(self.symbol):
                        send_tmp[i, 0] = self.send_signal[i, s]

                    receive_tmp += np.dot(H, send_tmp)

                for i in range(self.symbol):
                    self.receive_signal[i, r] = receive_tmp[i, 0]

        return self.receive_signal



