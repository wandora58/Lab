
import math
import numpy as np

from tqdm import tqdm

from Model.Selection import Selection
from Model.Convcode import Convcode
from Model.Modulation import Modulation
from Model.Sequence import ZadoffSequence, DFTSequence
from Model.Channel import Channel
from Model.Weight import Weight
from Model.Equalization import Equalization
from Model.Estimate import Estimate
from Model.utils import Bitrate, Serial2Parallel, Noise, Result, Parallel2Serial

def simulation(flame=1000, user=4, user_antenna=1, BS=1, BS_antenna=4, select_user=4, Nc=128, CP=1, path=1, Ps=1.0, data_symbol=128, M=4, conv_type='soft', code_rate=1/2, zad_len=16, zad_num=1, zad_shift=0, channel_type='Rayleigh', last_snr=30):

    code_symbol = int(data_symbol * 1/code_rate)
    bit_rate = Bitrate(M).count_bit_rate()
    selection = Selection(user, user_antenna, BS, BS_antenna, select_user)
    channel = Channel(user, user_antenna, BS_antenna, Nc, path, CP, Ps, code_symbol)
    convcode = Convcode(select_user, user_antenna, BS_antenna, data_symbol, bit_rate, code_rate, conv_type)
    modulation = Modulation(M, code_symbol, data_symbol, bit_rate, select_user, user_antenna, BS_antenna)
    equalization = Equalization(code_symbol, select_user, user_antenna, BS_anntena, Nc)

    for SNRdB in range(0,31):
        SNR = 10 ** (SNRdB / 10)
        No = 1 / SNR
        sigma = math.sqrt(No/2)
        true = Result(flame, user, BS, data_symbol, Nc)

        for count in tqdm(range(flame)):

            # Create channel
            true_channel = channel.create_rayleigh_channel()

            # User selection
            seleced_channel = selection.selection(true_channel)

            # Randomly create send bit with 0 and 1
            send_data = np.random.randint(0,2, select_user * user_antenna * data_symbol * bit_rate)

            # Serial to Parallel
            send_bit = Serial2Parallel(send_data, select_user, user_antenna, bit_rate, data_symbol).create_parallel()

            # Conv coding
            encode_bit = convcode.encoding(send_bit)

            # Modulation
            mod_signal = modulation.modulation(encode_bit)

            # Channel multiplication
            receive_signal = channel.channel_multiplication(seleced_channel, mod_signal)

            # Create noise
            noise = Noise().create_noise(code_symbol, sigma, BS_anntena)

            # Add noise
            receive_signal += noise

            # ZF equalization
            receive_weight_signal = equalization.MMSE(receive_signal, seleced_channel, SNR)

            # Demodulation
            demod_bit = modulation.demodulation(receive_weight_signal, conv_type, sigma)

            # Conv decoding
            decode_bit = convcode.decoding(demod_bit)

            # Parallel to Serial
            receive_data = Parallel2Serial(decode_bit).create_serial()

            # BER & NMSE & TRP
            true.calculate(count, send_data, receive_data, mod_signal, noise, SNR, M, bit_rate, code_rate)

            #---------------------------- zadoff -------------------------------------

            # # channel estimate
            # zadoff_channel = zadoff_estimate.least_square_estimate(true_channel)
            #
            # # Frequency domain equalization
            # zadoff_signal = equalization.ZF(receive_signal, zadoff_channel)
            #
            # # Demodulation
            # zadoff_demod_bit = modulation.demodulation(zadoff_signal, conv_type, sigma)
            #
            # # Conv decoding
            # zadoff_decode_bit = convcode.decoding(zadoff_demod_bit)
            #
            # # Parallel to Serial
            # zadoff_receive_data = Parallel2Serial(zadoff_decode_bit).create_serial()
            #
            # # BER & NMSE & TRP
            # zadoff.calculate(count, send_data, zadoff_receive_data, mod_signal, noise, SNR, M, bit_rate, code_rate, true_channel, zadoff_channel)


