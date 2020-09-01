
import numpy as np


class Selection:
    def __init__(self, user, user_antenna, BS, BS_antenna, select_user):
        self.user = user
        self.user_antenna = user_antenna
        self.BS = BS
        self.BS_antenna = BS_antenna
        self.select_user = select_user


    def CDUS(self, channel):

        U = range(self.user)
        S = []

        for k in range(self.select_user):
            if k == 0:
                s1 = 0
                H = np.zeros((self.user_antenna, self.BS_antenna), dtype=np.complex)
                for s in range(self.user):

                    for i in range(self.user_antenna):
                        for j in range(self.BS_antenna):
                            H[i, j] = channel[0, s*self.user_antenna*self.BS_antenna + i*self.BS_antenna + j]

                    tmp = np.linalg.norm(H,'fro')

                    if tmp > s1:
                        s1 = s
                        S1 = H

                U.remove(s1)
                S.append(s1)

            else:
                

        return [0,1,2,3]


