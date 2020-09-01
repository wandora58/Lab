
import numpy as np

from numpy.linalg import matrix_rank

from Model import utils


class Estimate:
    def __init__(self, pilot_signal, channel, user, BS, path, sigma, Nc):
        self.pilot_signal = pilot_signal
        self.channel = channel
        self.user = user
        self.BS = BS
        self.path = path
        self.sigma = sigma
        self.Nc = Nc


    def least_square_estimate(self):
        """
          Ax = b   A:(m,n)  b:(m,1)  x:(n,1)  m: Number of equations  n: Number of unknowns

          if rank(A) = m = n:  (A is square and full rank)
             x = A(-1)・b

          if rank(A) = n < m:  (A is vertically and column full rank)
             x = (A^T・A)^(-1)・A^T・b

          if rank(A) = m < n:  (A is horizontally and row full rank)
             x = A^T・(A・A^T)^(-1)・b

          if rank(A) < m, n;   (A is rank down)
             As rank(A) = r, factorize A(m,n) into B(m,r) and C(r,n)
             x = A^(-1)・b
               = C^(-1)・B^(-1)・b
               = C^T・(C・C^T)^(-1)・(B^T・B)^(-1)・B^T・b

        """
        estimated_channel = np.zeros((self.Nc, self.user*self.BS), dtype=np.complex)
        tmp_channel = np.zeros((self.user*self.path, 1),dtype=np.complex)
        pilot_len = len(self.pilot_signal)

        for r in range(self.BS):

            for s in range(self.user):
                for p in range(self.path):
                    tmp_channel[s*self.path + p,0] = self.channel[p, self.user*r+s]

            tmp_receive = np.dot(self.pilot_signal, tmp_channel)
            noise = utils.Noise().create_noise(pilot_len, self.sigma, 1)
            # tmp_receive += noise

            A_inv = np.linalg.pinv(self.pilot_signal)
            tmp = np.dot(A_inv, tmp_receive)

            for s in range(self.user):
                estimated_channel[0, self.user*r+s] = tmp[s, 0]

            # A = self.pilot_signal
            # Ah = np.conjugate(A.T)
            #
            # rank = np.linalg.matrix_rank(A)
            #
            # rank(A) = m < n → Rank decomposition
            # if rank < pilot_len:
            #     U, S, V = np.linalg.svd(A, full_matrices=True)
            #     Ur = U[:,:rank]
            #     Sr = np.diag(S[:rank])
            #     Vr = V[:rank,:]
            #     A_ = np.conjugate(Vr.T) @ np.linalg.inv(Sr) @ np.conjugate(Ur.T)
            #     tmp = np.conjugate(Vr.T) @ np.linalg.inv(Sr) @ np.conjugate(Ur.T) @ tmp_receive
            #
            # rank(A) = m < n
            # elif rank == pilot_len and rank < self.user:
            #     tmp = Ah @ np.linalg.inv(A @ Ah) @ tmp_receive
            #
            # rank(A) = m = n
            # else:
            #     tmp = np.linalg.inv(Ah @ A) @ Ah @ tmp_receive

        return estimated_channel, tmp_receive
