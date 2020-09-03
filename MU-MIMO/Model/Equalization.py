
import numpy as np

from scipy.linalg import dft


class Equalization:
    def __init__(self, code_symbol, user, user_antenna, BS_anntena, Nc):
        self.symbol = code_symbol
        self.user = user
        self.user_antenna = user_antenna
        self.BS = BS_anntena
        self.Nc = Nc


    def ZF(self, receive_signal, channel):
        """

        ZF Equalization

        Args:
            receive_signal : 2D ndarray [BS, symbol * 1/code_rate]

                   chennel : 2D ndarray [Nc, user * BS]


        Returns:
                 ZF_signal : 2D ndarray [user, symbol * 1/code_rate]

                               H  =  2D ndarray [BS, user]

                                                 user
                                       BS1 [[h11 h21 h31 ..],
                                       BS2  [h12 h22 h32 ..],
                                            [      ...     ]]


                       Weight_ZF  =  2D ndarray [user, BS]

                                        (H^H * H)^-1  *  H^H
                                        [user, user]   [user, BS]


                        ZF_signal =  2D ndarray [user, symbol * 1/code_rate]

                                       for i in symbol:
                                          R  =  2D ndarray [BS, 1]

                                                  BS1 [[symbol1i]
                                                  BS2  [symbol2i]
                                                       [  ...   ]]

                                          for s in user:
                                              ZF_signal[s,i]  =  Weight_ZF  *  R
                                                                [user, BS]    [BS, 1]


        """
        self.receive_signal = receive_signal
        self.channel = channel

        H = np.zeros((self.BS, self.user), dtype=np.complex)
        R = np.zeros((self.BS, 1), dtype=np.complex)
        ZF_signal = np.zeros((self.user, self.symbol), dtype=np.complex)

        for r in range(self.BS):
            for s in range(self.user):
                H[r][s] = self.channel[0][self.BS*r + s]

        # Weight_ZF = np.linalg.pinv(np.conjugate(H.T) @ H) @ np.conjugate(H.T)
        Weight_ZF = np.linalg.pinv(H)

        for i in range(self.symbol):
            R[:,0] = self.receive_signal[:,i]
            S = Weight_ZF @ R

            for s in range(self.user):
                ZF_signal[s,i] = S[s,0]

        return np.round(ZF_signal, decimals=7)


    def MMSE(self, receive_signal, seleced_channel, SNR):
        """

        MMSE Equalization

        Args:
            receive_signal : 2D ndarray [BS_antenna, symbol * 1/code_rate]
           seleced_channel : 2D ndarray [BS_antenna, select_user * user_antenna]

        Returns:
               MMSE_signal : 2D ndarray [select_user * user_antenna, symbol * 1/code_rate]

                                H = seleced_channel
                               Nt = user * user_antenna
                               Nr = BS_anntena


                      Weight_MMSE = (H^H * H + (Nt * INt)/SNR )^-1  *  H^H

                      for i in symbol:
                          S: 2D ndarray [user * user_antenna, 1]
                          R: 2D ndarray [BS_anntena, 1]

                          S = Weight_MMSE * R

                          for s in user * user_antenna:
                              MMSE_signal[s,i] = S[s,0]


        """
        H = seleced_channel
        Nt = self.user * self.user_antenna
        INt = (Nt / SNR) * np.eye(Nt)

        Weight_MMSE = np.linalg.pinv(np.conjugate(H.T) @ H + INt) @ np.conjugate(H.T)

        R = np.zeros((self.BS, 1), dtype=np.complex)
        MMSE_signal = np.zeros((self.user * self.user_antenna, self.symbol), dtype=np.complex)

        for i in range(self.symbol):
            R[:,0] = receive_signal[:,i]
            S = Weight_MMSE @ R

            for s in range(self.user * self.user_antenna):
                MMSE_signal[s,i] = S[s,0]

        return np.round(MMSE_signal, decimals=7)



    def MMSE_FDE(self, receive_signal, channel, sigma):
        self.receive_signal = receive_signal
        self.channel = channel
        self.sigma = sigma

        # DFT matrix
        DFT = dft(self.Nc)
        IDFT = (np.conjugate(DFT).T) / self.Nc

        # Fourier transform
        Fourier_channel = np.dot(DFT, self.channel)
        Fourier_receive_signal = np.dot(DFT, self.receive_signal)

        # Noise Power
        Pn = 2 * self.sigma * self.sigma * self.user

        # Eye matrix for MMSE
        I = np.eye(self.BS, dtype=np.complex)
        I = Pn * I

        # Calculate MMSE Weight
        H = np.zeros((self.BS, self.user), dtype=np.complex)
        Fourier_FDE_signal = np.zeros((self.Nc, self.user), dtype=np.complex)
        R = np.zeros((self.BS, 1), dtype=np.complex)

        for r in range(self.BS):
            for s in range(self.user):
                H[r, s] = Fourier_channel[0, self.BS * r + s]

        MMSE_W = np.dot(H, np.conjugate(H.T)) + I
        MMSE_Weight = np.dot(np.linalg.inv(MMSE_W), H)

        for i in range(self.Nc):

            for r in range(self.BS):
                R[r, 0] = Fourier_receive_signal[i, r]

            tmp = np.dot(np.conjugate(MMSE_Weight.T), R)

            for s in range(self.user):
                Fourier_FDE_signal[i, s] = tmp[s, 0]

        # Inv Fourier transform
        FDE_signal = np.dot(IDFT,Fourier_FDE_signal)

        return FDE_signal

