
import math
import numpy as np

from numpy.linalg import svd

class Weight:
    def __init__(self, user, BS):
        self.user = user
        self.BS = BS


    def send_weight_multiplication(self, send_signal, channel, num_stream, send_power):

        self.num_stream = num_stream
        self.send_power = send_power

        self.H = np.zeros((self.BS, self.user), dtype=np.complex)
        for r in range(self.BS):
            for s in range(self.user):
                self.H[r][s] = channel[0][self.BS*r + s]

        gram_H = np.conjugate(self.H.T) @ self.H
        ramda, eig_vec = np.linalg.eig(gram_H)

        self.Ak = np.diag(ramda[:self.num_stream])
        self.Uk = eig_vec[:,:self.num_stream]

        Pk = np.diag([math.pow(i, 1/2) for i in self.send_power])

        weight_signal = self.Uk @ Pk @ send_signal

        return weight_signal


    def receive_weight_multiplication(self, receive_signal):

        Pk = np.diag([math.pow(i, -1/2) for i in self.send_power])
        weight_signal = np.linalg.inv(self.Ak) @ Pk @ np.conjugate(self.Uk.T) @ np.conjugate(self.H.T) @ receive_signal

        return weight_signal


