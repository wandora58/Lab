import math
import numpy as np
from Model.Channel import Channel
from numpy.linalg import svd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cumulative_probability(channel, user, BS):
    count = 100000
    matrix = np.zeros((count,user), dtype=np.float)

    for cnt in range(count):
        h = channel.create_rayleigh_channel()
        H = np.zeros((BS, user), dtype=np.complex)

        for r in range(BS):
            for s in range(user):
                H[r][s] = h[0][BS*r + s]

        gram_H = np.conjugate(H.T) @ H
        U, A ,Uh = svd(gram_H)

        for s in range(user):
            matrix[cnt][s] = 10 * math.log10(A[s])

    matrix = np.sort(matrix,axis=0)

    x = range(-40,20)
    cumulative_matrix = np.zeros((user, len(x)),dtype=np.double)

    for i in range(len(x)) :
        for s in range(user):
            tmp = np.where(matrix[:,s] <= x[i], 1, matrix[:,s])
            cumulative_matrix[s,i] = sum(tmp == 1) / count

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(user):
        ax.plot(x, cumulative_matrix[i], linestyle='-', color='k')

    plt.gca().xaxis.set_tick_params(direction='in')
    plt.gca().yaxis.set_tick_params(direction='in')

    ax.set_xlabel('eigenvalue [dB]', fontsize=12)
    ax.set_ylabel('Cumulative probability', fontsize=12)

    plt.grid(which="both")

    plt.savefig('Image/CDF/user{}_BS{}.pdf'.format(user, BS))

    #-----------NMSE--------

symbol = 128
user = 4
BS = 4
Nc = 128
path = 1
CP = 1
Ps = 1

channel = Channel(symbol, user, BS, Nc, path, CP, Ps)

cumulative_probability(channel, user, BS)