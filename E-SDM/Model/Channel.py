
import numpy as np
import math

from Model import utils


class Channel:
    def __init__(self, user, BS, Nc, path, CP, Ps):
        """

        Mimo channel class.

        Args:
            symbol (int): the number of symbol
            user (int): the number of user
            BS (int): the number of BS
            Nc (int): the number of carriers
            Path (int): The number of path
            CP (int): The number of cyclic prefix
            Ps (float): Total received power
            weight (float): The weight

        """
        self.user = user
        self.BS = BS
        self.Nc = Nc
        self.path = path
        self.CP = CP
        self.Ps = Ps
        self.weight = utils.Weight().create_weight(path)
        self.channel = np.zeros((Nc, user * BS), dtype=np.complex)


    def create_rayleigh_channel(self):
        """

        Create Rayleigh fading channel

        Returns:
            channel (np.array): Rayleigh fading channel with 1 dB attenuation by path

                                            BS1           BS2
                                         u1  u2 ... /  u1 us2 ... /
                               path1 [[ h11 h21 h31   h12 h22 h32 ],
                               path2  [ h11 h21 h31   h12 h22 h32 ],                          ]
                               path3  [ h11 h21 h31   h12 h22 h32 ],
                                                   ...            ]]

                                  hij : Channel from i-th user to j-th BS
                                        1db attenuation for each additional path


        """
        for i in range(self.user * self.BS):

            Ps = self.Ps
            for path in range(self.path):
                self.channel[path, i] = utils.Boxmuller().create_normalized_random(Ps / self.weight * math.sqrt(0.5))
                # Ps *= math.sqrt(math.pow(10, -0.1))  # 1dB減衰

        return self.channel


    def channel_multiplication(self, send_signal, code_symbol):
        """

        Multiplies send signal by mimo channel

        Args:
            send_signal (np.array): send signal (us, symbol)

                                    us1  [[symbol11, symbol12, symbol13]
                                    us2   [symbol21, symbol22, symbol23]
                                    us3   [symbol31, symbol32, symbol33]
                                          [  ...  ,    ...   ,   ...   ]]

                                      symbolij : j-th symbol of i-th user

        Returns:
            receive_signal (np.array): Received signal after multiplying send signal by channel

                                           bs1  [[symbol11, symbol12, symbol13]
                                           bs2   [symbol21, symbol22, symbol23]
                                           bs3   [symbol31, symbol32, symbol33]
                                                 [  ...  ,    ...   ,   ...   ]]

                                       If path == 1:
                                          channel is converted to a diagonal matrix and multiplied by send signal

                                       If path >= 1:
                                          channel is processed by Cyclic Prefix and then send signal is multiplied

        """
        self.symbol = code_symbol
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



