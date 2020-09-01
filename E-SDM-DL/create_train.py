
import csv
import math, cmath
import random
import numpy as np
from tqdm import tqdm
from numpy.linalg import svd

from Model.BERControl import BERControl
from Model.Convcode import Convcode
from Model.Modulation import Modulation
from Model.Sequence import ZadoffSequence, DFTSequence
from Model.Frame import Frame
from Model.Channel import Channel
from Model.Weight import Weight
from Model.utils import Serial2Parallel, Noise, Result, Parallel2Serial


def conv_file_name(rate):
    if rate == 1/2:
        return '1/2'

    elif rate == 2/3:
        return '2/3'

    elif rate == 3/4:
        return '3/4'

if __name__ == '__main__':

    count = 2
    is_random = True
    SNR = 20
    user = 4

    data_symbol = 72
    BS = user
    Nc = 72
    path = 1
    CP = 1
    Ps = 1
    total_bit = user*2

    code_rates = [1/2, 2/3, 3/4]
    frame = 2

    conv_type = 'soft'
    BW = 1/6

    bit_dict = []

    convcode = Convcode(user, BS, data_symbol, conv_type)
    modulation = Modulation(user, BS)
    channel = Channel(user, BS, Nc, path, CP, Ps)
    weight = Weight(user, BS)

    for cnt in tqdm(range(count)):
        if is_random == True:
            SNRdB = random.randint(20, 25)

        SNR = 10 ** (SNRdB / 10)
        No = 1 / SNR
        sigma = math.sqrt(No/2)

        true_channel = channel.create_rayleigh_channel()
        H = np.zeros((BS, user), dtype=np.complex)
        for r in range(BS):
            for s in range(user):
                H[r][s] = true_channel[0][BS*r + s]
        gram_H = np.conjugate(H.T) @ H
        lamda, eig_vec = np.linalg.eig(gram_H)

        num_stream, bit_rate, send_power, code_rate = BERControl(user, BS, true_channel, sigma, Ps, total_bit).bit_allocation()
        TRP = []
        for code_rate in code_rates:
            s = 0
            n = 0
            rrs = 0
            rrn = 0
            ber = 0
            bler = 0

            for i in tqdm(range(frame)):

                # Randomly create send bit with 0 and 1
                send_data = np.random.randint(0,2,data_symbol*total_bit)

                # Serial to Parallel
                send_bit = Serial2Parallel(send_data, bit_rate, data_symbol).create_parallel()

                # Conv coding
                encode_bit, code_symbol = convcode.encoding(send_bit, num_stream, bit_rate, code_rate)

                # Modulation
                send_signal = modulation.modulation(encode_bit, num_stream, bit_rate, code_symbol)

                # Send weight multiplication
                send_weight_signal = weight.send_weight_multiplication(send_signal, true_channel, num_stream, send_power)

                # Channel multiplication
                receive_signal = channel.channel_multiplication(send_weight_signal, code_symbol)

                # Create noise
                noise = Noise().create_noise(code_symbol, sigma, BS)

                # Add noise
                receive_signal += noise

                # receive weight multiplication
                receive_weight_signal = weight.receive_weight_multiplication(receive_signal)

                # Demodulation
                demod_bit = modulation.demodulation(receive_weight_signal, conv_type, sigma)

                # Conv decoding
                decode_bit = convcode.decoding(demod_bit)

                # Parallel to Serial
                receive_data = Parallel2Serial(decode_bit, bit_rate, data_symbol).create_serial()

                t = True
                for i in range(data_symbol*total_bit):
                    if send_data[i] != receive_data[i]:
                        ber += 1
                        t = False

                if t == False:
                    bler += 1

                for i in range(code_symbol):
                    rrs += np.sum(np.abs(send_weight_signal) * np.abs(send_weight_signal))
                    rrn += np.sum(np.abs(noise) * np.abs(noise))

            ber /= frame * data_symbol * total_bit
            bler /= frame
            print('-----------------------------------------------')
            print('   BIT_RATE : ', bit_rate)
            print('    SNR[dB] : ', SNRdB)
            # print(' AveSNR[dB] : ', 10 * math.log10(rrs/rrn))
            print('        BER : ', ber)
            print('-----------------------------------------------')

            trp = 0
            for bit in bit_rate:
                trp += data_symbol * bit * code_rate * (1-bler)
            TRP.append(trp)

        print('        TRP : ', TRP)
        print('-----------------------------------------------')

        l1 = [0 for i in range(round(total_bit/8))]
        l2 = [0 for i in range(round(total_bit/6))]
        l3 = [0 for i in range(round(total_bit/4))]
        l4 = [0 for i in range(round(total_bit/2))]

        k1 = 0
        k2 = 0
        k3 = 0
        k4 = 0
        for i in bit_rate:
            if i == 8:
                l1[k1] = 1
                k1 += 1

            if i == 6:
                l2[k2] = 1
                k2 += 1

            if i == 4:
                l3[k3] = 1
                k3 += 1

            if i == 2:
                l4[k4] = 1
                k4 += 1

        lst = l1 + l2 + l3 + l4

        real_h = np.real(true_channel[0][:])
        imag_h = np.imag(true_channel[0][:])

        inp = []
        for i in range(len(true_channel[0][:])):
            inp.append(real_h[i])
            inp.append(imag_h[i])

        if is_random == True:
            onehot_bit_file = 'Data/train_data/onehot_bit/user{}_bit{}_SNR={}.csv'.format(user, total_bit, 'random')
            bit_file = 'Data/train_data/bit/user{}_bit{}_SNR={}.csv'.format(user, total_bit, 'random')
            channel_file = 'Data/train_data/channel/user{}_bit{}_SNR={}.csv'.format(user, total_bit, 'random')
            conv_rate_file = 'Data/train_data/conv_rate/user{}_bit{}_SNR={}.csv'.format(user, total_bit, 'random')
            eigen_file = 'Data/train_data/eigen/user{}_bit{}_SNR={}.csv'.format(user, total_bit, 'random')
            sigma_file = 'Data/train_data/sigma/user{}_bit{}_SNR={}.csv'.format(user, total_bit, 'random')

        else:
            onehot_bit_file = 'Data/train_data/onehot_bit/user{}_bit{}_SNR={}.csv'.format(user, total_bit, SNR)
            bit_file = 'Data/train_data/bit/user{}_bit{}_SNR={}.csv'.format(user, total_bit, 'random')
            channel_file = 'Data/train_data/channel/user{}_bit{}_SNR={}.csv'.format(user, total_bit, SNR)
            conv_rate_file = 'Data/train_data/conv_rate/user{}_bit{}_SNR={}.csv'.format(user, total_bit, SNR)
            eigen_file = 'Data/train_data/eigen/user{}_bit{}_SNR={}.csv'.format(user, total_bit, SNR)
            sigma_file = 'Data/train_data/sigma/user{}_bit{}_SNR={}.csv'.format(user, total_bit, SNR)

        if cnt == 0:
            with open(onehot_bit_file, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(lst)

            with open(bit_file, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(bit_rate)

            with open(channel_file, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(inp)

            with open(conv_rate_file, mode='w') as f:
                f.write(conv_file_name(code_rates[TRP.index(max(TRP))]) + '\n')

            with open(eigen_file, mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(np.real(lamda))

            if is_random == True:
                with open(sigma_file, mode='w') as f:
                    f.write(str(sigma) + '\n')

        else:
            with open(onehot_bit_file, mode='a') as f:
                writer = csv.writer(f)
                writer.writerow(lst)

            with open(bit_file, mode='a') as f:
                writer = csv.writer(f)
                writer.writerow(bit_rate)

            with open(channel_file, mode='a') as f:
                writer = csv.writer(f)
                writer.writerow(inp)

            with open(conv_rate_file, mode='a') as f:
                f.write(conv_file_name(code_rates[TRP.index(max(TRP))]) + '\n')

            with open(eigen_file, mode='a') as f:
                writer = csv.writer(f)
                writer.writerow(np.real(lamda))

            if is_random == True:
                with open(sigma_file, mode='a') as f:
                    f.write(str(sigma) + '\n')









