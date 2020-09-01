
import math
import numpy as np

from tqdm import tqdm

from Model.Convcode import Convcode
from Model.Modulation import Modulation
from Model.Sequence import ZadoffSequence, DFTSequence
from Model.Frame import Frame
from Model.Channel import Channel
from Model.Equalization import Equalization
from Model.Estimate import Estimate
from Model.utils import Bitrate, Serial2Parallel, Parallel2Serial, Noise, Result


def simulation(flame=1000, user=16, BS=1, Nc=128, CP=1, path=1, Ps=1.0, BW=80, data_symbol=100, Ms=4, code_rates=1/2, conv_type='hard', zad_len=4, zad_num=0, zad_shift=1, channel_type='Rayleigh', csv_column=False, last_snr=30):


    for SNR in range(0,last_snr):

        sigma = math.sqrt( Ps / (2 * math.pow(10.0, SNR/10)))
        zadoff_pilot = ZadoffSequence(user, path, zad_len, zad_num, zad_shift).create_pilot()
        zadoff_estimate = Estimate(zadoff_pilot, user, BS, path, sigma, Nc)

        # dft_pilot = DFTSequence(user, path, pilot_len, size).create_pilot()
        # frame_pilot = Frame(user, pilot_len).create_frame_matrix()

        true = Result(flame, user, BS, BW, data_symbol, Nc, conv_type, channel_type, csv_column, last_snr)
        zadoff = Result(flame, user, BS, BW, data_symbol, Nc, conv_type, channel_type, csv_column, last_snr)

        for code_rate in code_rates:
            for M in Ms:

                code_symbol = int(data_symbol * 1/code_rate)
                bit_rate = Bitrate(M).count_bit_rate()
                convcode = Convcode(user, BS, data_symbol, bit_rate, code_rate, conv_type)
                modulation = Modulation(M, code_symbol, data_symbol, bit_rate, user, BS)
                channel = Channel(code_symbol, user, BS, Nc, path, CP, Ps, channel_type)
                equalization = Equalization(code_symbol, user, BS, Nc)

                for count in tqdm(range(flame)):

                    # Randomly create send bit with 0 and 1
                    send_data = np.random.randint(0,2,user*data_symbol*bit_rate)

                    # Serial to Parallel
                    send_bit = Serial2Parallel(send_data, user, bit_rate, data_symbol).create_parallel()

                    # Conv coding
                    encode_bit = convcode.encoding(send_bit)

                    # Modulation
                    mod_signal = modulation.modulation(encode_bit)

                    # Create channel
                    true_channel = channel.create_channel()

                    # Channel multiplication
                    receive_signal = channel.channel_multiplication(mod_signal)

                    # Create noise
                    noise = Noise().create_noise(code_symbol, sigma, BS)

                    # Add noise
                    receive_signal += noise

                    # ZF equalization
                    receive_weight_signal = equalization.ZF(receive_signal, true_channel)

                    # Demodulation
                    demod_bit = modulation.demodulation(receive_weight_signal, conv_type, sigma)

                    # Conv decoding
                    decode_bit = convcode.decoding(demod_bit)

                    # Parallel to Serial
                    receive_data = Parallel2Serial(decode_bit).create_serial()

                    # BER & NMSE & TRP
                    true.calculate(count, send_data, receive_data, mod_signal, noise, SNR, M, bit_rate, code_rate)

                    #---------------------------- zadoff -------------------------------------

                    # channel estimate
                    zadoff_channel = zadoff_estimate.least_square_estimate(true_channel)

                    # Frequency domain equalization
                    zadoff_signal = equalization.ZF(receive_signal, zadoff_channel)

                    # Demodulation
                    zadoff_demod_bit = modulation.demodulation(zadoff_signal, conv_type, sigma)

                    # Conv decoding
                    zadoff_decode_bit = convcode.decoding(zadoff_demod_bit)

                    # Parallel to Serial
                    zadoff_receive_data = Parallel2Serial(zadoff_decode_bit).create_serial()

                    # BER & NMSE & TRP
                    zadoff.calculate(count, send_data, zadoff_receive_data, mod_signal, noise, SNR, M, bit_rate, code_rate, true_channel, zadoff_channel)
                    #-------------------------------------------------------------------------


