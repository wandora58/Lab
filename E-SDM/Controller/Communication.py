
import math
import numpy as np

from tqdm import tqdm

from Model.BERControl import BERControl
from Model.Convcode import Convcode
from Model.Modulation import Modulation
from Model.Sequence import ZadoffSequence, DFTSequence
from Model.Frame import Frame
from Model.Channel import Channel
from Model.Weight import Weight
from Model.Equalization import Equalization
from Model.Estimate import Estimate
from Model.utils import Serial2Parallel, Noise, Result, Parallel2Serial

def simulation(flame=1000, user=16, BS=1, Nc=128, CP=1, path=1, Ps=1.0, BW=1/6, data_symbol=128, conv_type='soft', total_bit=8, zad_len=16, zad_num=1, zad_shift=0, channel_type='Rayleigh', last_snr=30):

    convcode = Convcode(user, BS, data_symbol, conv_type)
    modulation = Modulation(user, BS)
    channel = Channel(user, BS, Nc, path, CP, Ps)
    weight = Weight(user, BS)

    for SNR in range(0,31):
        sigma = math.sqrt( Ps / (2 * math.pow(10.0, SNR/10)))
        true = Result(flame, user, BS, data_symbol, Nc)

        for count in tqdm(range(flame)):

            # Create channel
            true_channel = channel.create_rayleigh_channel()

            # Bit allocation & Send power control
            num_stream, bit_rate, send_power, code_rate = BERControl(user, BS, true_channel, sigma, Ps, total_bit).bit_allocation()

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

            # BER
            true.calculate(count, send_data, receive_data, send_weight_signal, noise, SNR, total_bit)

