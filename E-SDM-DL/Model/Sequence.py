
import numpy as np
import math


class Sequence:
    def __init__(self, user, path, pilot_len):
        """

        Sequence class.

        Args:
            user (int): the number of user
            path (int): The number of path
            pilot_len (int): The length of pilot

        """
        self.user = user
        self.path = path
        self.pilot = np.zeros((pilot_len, 1), dtype=np.complex)
        self.pilot_signal = np.zeros((pilot_len, user*path), dtype=np.complex)


class MSequence(Sequence):
    def __init__(self, user, path, m_num, m_len):
        """

        M-Sequence class.

        Args:
            m_num (int): the number of M-sequence
            m_len (int): The length of M-sequence

        """
        super().__init__(user, path, m_len)
        self.pilot = self.pilot.astype(np.int)
        self.pilot_signal = self.pilot_signal.astype(np.int)
        self.num = m_num
        self.len = m_len

    def create_sequence(self):
        if self.num == 4 and self.len == 15:

            self.pilot[0] = 1
            self.pilot[3] = 1

            for i in range(self.num, self.len):
                self.pilot[i] = self.pilot[i - 4] ^ self.pilot[i - 1]

        else:
            raise ValueError("M error!")

        return self.pilot

    def create_pilot(self):
        """

        Create M-sequence pilot signal

        Returns:
            pilot_signal (np.array): send same signal to different paths for the same user

                                               u1                    u2
                                        path1 path2   ...  /  path1 path2   ... /
                               time1 [[   s11   s11   s11       s21   s21   s21 ],
                    pilot_len  time2  [   s12   s12   s12       s22   s22   s22 ],                          ]
                               time3  [   s13   s13   s13       s23   s23   s23 ],
                                      [                      ...                ]]

                                  sij : Signal sent by the i-th user at j-time j

        """

        self.create_sequence()

        for s in range(self.user):
            shift = np.roll(self.pilot, s)
            for i in range(self.len):
                for j in range(self.path):
                    self.pilot_signal[i, self.path*s + j] = shift[i, 0]

        return self.pilot_signal


class GoldSequence(MSequence):
    def __init__(self, user, path, gold_num, gold_len):
        super().__init__(user, path, gold_num, gold_len)
        self.pilot = self.pilot.astype(np.int)

    def create_sequence(self):
        m1 = np.zeros((self.len, 1), dtype=np.int)
        m2 = np.zeros((self.len, 1), dtype=np.int)

        if self.num == 4 and self.len == 15:

            m1[0] = 1
            m1[3] = 1

            m2[0] = 1
            m2[2] = 1
            m2[3] = 1

            for i in range(self.num, self.len):
                m1[i] = m1[i - 4] ^ m1[i - 1]
                m2[i] = m2[i - 4] ^ m2[i - 2]

            for j in range(self.len):
                self.pilot[j] = m1[j] ^ m2[j]

        elif self.num == 5 and self.len == 31:

            m1[2] = 1
            m1[4] = 1

            m2[0] = 1
            m2[2] = 1
            m2[4] = 1

            for i in range(self.num, self.len):
                m1[i] = m1[i - 5] ^ m1[i - 2]
                m2[i] = m2[i - 5] ^ m2[i - 4] ^ m2[i - 3] ^ m2[i - 2]

            for j in range(self.len):
                self.pilot[j] = m1[j] ^ m2[j]

        elif self.num == 6 and self.len == 63:

            m1[0] = 1
            m1[5] = 1

            m2[0] = 1
            m2[3] = 1
            m2[5] = 1

            for i in range(self.num, self.len):
                m1[i] = m1[i - 6] ^ m1[i - 1]
                m2[i] = m2[i - 6] ^ m2[i - 5] ^ m2[i - 2] ^ m2[i - 1]

            for j in range(self.len):
                self.pilot[j] = m1[j] ^ m2[j]

        elif self.num == 7 and self.len == 127:

            m1[6] = 1

            m2[4] = 1
            m2[6] = 1

            for i in range(self.num, self.len):
                m1[i] = m1[i - 7] ^ m1[i - 3]
                m2[i] = m2[i - 7] ^ m2[i - 3] ^ m2[i - 2] ^ m2[i - 1]

            for j in range(self.len):
                self.pilot[j] = m1[j] ^ m2[j]

        else:
            raise ValueError("Gold error!")

        return self.pilot

    def create_pilot(self):

        self.create_sequence()

        for s in range(self.user):
            shift = np.roll(self.pilot, s)
            for i in range(self.len):
                for j in range(self.path):
                    self.pilot_signal[i, self.path*s + j] = shift[i, 0]

        return self.pilot_signal


class ZadoffSequence(Sequence):
    def __init__(self, user, path, zad_len, zad_num, zad_shift):
        super().__init__(user, path, zad_len)
        self.len = zad_len
        self.num = zad_num
        self.shift = zad_shift

    def create_sequence(self):

        for k in range(self.len):
            n1 = (k + self.shift) % self.len
            self.pilot[k, 0] = np.exp(-1j * math.pi * self.num * n1 * (n1) / self.len)

        return self.pilot

    def create_pilot(self):

        for s in range(self.user):
            tmp = self.create_sequence()
            self.shift += self.path

            for i in range(self.len):
                for j in range(self.path):
                    self.pilot_signal[i, self.path*s +j] = tmp[i, 0]

        return self.pilot_signal


class DFTSequence(Sequence):
    def __init__(self, user, path, pilot_len, size):
        super().__init__(user, path, pilot_len)
        self.pilot_len = pilot_len
        self.size = size

    def create_pilot(self):

        for s in range(self.user):
            for j in range(self.pilot_len):
                self.pilot_signal[j, s] = np.exp((-1j * 2 * math.pi * s * j) / self.size)

        return self.pilot_signal

