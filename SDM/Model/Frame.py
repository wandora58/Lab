
import numpy as np
import math
import cmath


class Frame:
    def __init__(self, N, m):
        """

        Frame matrix

        Args:
            N (int): number of frame vectors (user)
            m (int): number of frame dimensions (pilot_len)

        """
        self.N = N
        self.m = m


    def is_prime(self, q):
        """

        Prime judjement

        Return: (bool) if q == prime: True
                          q != prime: False
        """

        q = abs(q)
        if q == 2:
            return True

        if q < 2 or q & 1 == 0:
            return False

        return pow(2, q-1, q) == 1


    def is_divide(self, p, q):
        """

        divide judjement

        Return: (bool)  if p % q == 0: True
                           p % q != 0: False
        """

        if p % q == 0:
            return True

        else:
            return False


    def find_subgroups(self, n, m):

        """
        find cyclic group with m elements from the multiple group modulo n

        ex) n = 37, m = 9

          n is prime, rank = 36

          Z37* = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
                  25,26,27,28,29,30,31,32,33,34,35,36}

          if a = 7, rank = 9

            7^1 mod 37 = 7, 7^2 mod 37 = 14, .... 7^9 mod 37 = 1

              A = {1, 7, 9, 10, 12, 16, 26, 33, 34}
             2A = {2, 14, 15, 18, 20, 24, 29, 31, 32}
             3A = {3, 4, 11, 21, 25, 27, 28, 30, 36}
             5A = {5, 6, 8, 13, 17, 19, 22, 23, 35}

        """
        for i in range(2, n-1):

            lst = []
            cnt = 0

            while True:
                tmp = (i ** (cnt+1)) % n
                # tmp = (2*i ** (cnt+1)) % n

                if tmp in lst or len(lst) > m:
                    break
                else:
                    lst.append(tmp)

                cnt += 1

            if len(lst) == m:
                return sorted(lst)



    def find_uniques(self, A):
        l = []
        for i in range(self.N):
            for j in range(self.N):
                if round(A[i][j], 8) not in l:
                    l.append(round(A[i][j], 8))

        print(l)


    def create_frame_matrix(self):
        """

        1, Let n: prime, m: divisor of n−1, and G: (Z/Zn) multiple group modulo n

        2, prepare K: cyclic group such as subgroup of G with m elements

        3, prepare: w = exp(2πj/n), v = 1/sqrt(m)[1,,,1]^T, U = diag(w^(k1),...,w^(km))

        4, create F = [v, Uv, ... , U^(n-1)v] that have (n-1)/m different dot products

        return: Unit norm tight frame
                shape: (pilot_len, user)

        """

        if not self.is_prime(self.N):
            raise PrimeError('N is not prime')

        if not self.is_divide(self.N-1, self.m):
            raise DivideError('N-1 is not divisible by m')

        n = self.N

        if not self.is_divide(n-1, self.m):
            raise DivideError('n-1 is not divisible by m')

        else:
            K = self.find_subgroups(n,self.m)
            r = int((n-1)/self.m)

        v = np.full((self.m, 1), 1/math.sqrt(self.m))
        F = np.zeros((self.m, self.N), dtype=np.complex)

        for i in range(self.N):
            U = np.diag([cmath.exp(2j * math.pi * k * i/ n) for k in K])
            tmp = np.dot(U , v)

            for j in range(self.m):
                F[j,i] = tmp[j,0]

        # A = np.conjugate(F.T) @ F
        # self.find_uniques(A)


        return F


class PrimeError(Exception):
    pass

class DivideError(Exception):
    pass

