
import math
import cmath
import numpy as np

from numpy.linalg import svd


class BERControl:
    def __init__(self, user, BS, channel, sigma, Ps, total_bit):
        """

        Transmission rate and transmission power control based on BER

        """
        self.user = user
        self.BS = BS
        self.channel = channel
        self.sigma = sigma
        self.Ps = Ps
        self.Pn = 2 * sigma * sigma
        self.total_bit = total_bit
        self.combination = []


    def subset_sum(self, numbers, target, partial=[]):
        s = sum(partial)

        # check if the partial sum is equals to target
        if s == target and partial not in self.combination:
            self.combination.append(partial)

        # if we reach the number why bother to continue
        if s >= target:
            return

        if partial in self.combination:
            return

        for i in range(len(numbers)):
            n = numbers[i]
            remaining = numbers[i+1:]
            self.subset_sum(remaining, target, partial + [n])

        return self.combination


    def calculate_send_power(self, bits, ramda, r0, a, b):
        num_stream = len(bits)
        over = 0
        under = 0
        for k in range(num_stream):
            rk = ramda[k] * r0
            ak = a[bits[k]]
            bk = b[bits[k]]
            mk = bits[k]

            over += bk/rk * cmath.log(ak * mk * rk / bk)
            under += bk/rk

        xi = (over - 1) / under
        p = []

        for k in range(num_stream):
            rk = ramda[k] * r0
            ak = a[bits[k]]
            bk = b[bits[k]]
            mk = bits[k]

            if bk/rk * (cmath.log(ak * mk * rk / bk) - xi) > 0:
                p.append(np.real(bk/rk * (cmath.log(ak * mk * rk / bk) - xi)))

        if len(p) != num_stream:
            p = self.calculate_send_power(bits[:-1], ramda, r0, a, b)

        return p


    def bit_allocation(self):
        H = np.zeros((self.BS, self.user), dtype=np.complex)

        for r in range(self.BS):
            for s in range(self.user):
                H[r][s] = self.channel[0][self.BS*r + s]

        gram_H = np.conjugate(H.T) @ H
        ramda, eig_vec = np.linalg.eig(gram_H)

        r0 = self.Ps/self.Pn

        a = [0,0,1/2,0,3/8,0,7/24,0,15/64]
        b = [0,0,2,0,10,0,42,0,170]

        l1 = [8 for i in range(round(self.total_bit/8))]
        l2 = [6 for i in range(round(self.total_bit/6))]
        l3 = [4 for i in range(round(self.total_bit/4))]
        l4 = [2 for i in range(round(self.total_bit/2))]

        bit_pattern = self.subset_sum(l1+l2+l3+l4, self.total_bit)
        Peb = []
        send_power = []

        for bits in bit_pattern:
            num_stream = len(bits)
            tmp = 0
            p = self.calculate_send_power(bits, ramda, r0, a, b)
            if len(p) != num_stream:
                for i in range(num_stream - len(p)):
                    p.append(0)

            for k in range(num_stream):
                rk = ramda[k] * r0
                ak = a[bits[k]]
                bk = b[bits[k]]
                mk = bits[k]
                pk = p[k]

                tmp += 1/self.user * ak * mk * cmath.exp(-1 * pk * rk / bk)
            Peb.append(abs(tmp))
            send_power.append(p)

        idx = Peb.index(min(Peb))

        num_stream = len(bit_pattern[idx])
        code_rate = 1/2

        return num_stream, np.array(bit_pattern[idx]), send_power[idx], code_rate

