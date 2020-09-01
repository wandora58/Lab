
import numpy as np

from Controller import Communication


def main():
    flame = 2 #シミュレーション回数

    user = 4 # ユーザー数
    BS = 4 # 基地局数
    Nc = 128 # 搬送波数

    CP = 1 # CP数
    path = 1 # パス数
    Ps = 1.0 # 総送信電力

    BW = 1/6 # チャネル帯域幅[Mhz]
    data_symbol = 12 # 1フレーム内のシンボル数

    zad_len = 4 # Zadoff_Chu系列長　
    zad_num = 1 # Zadoff_Chu系列番号
    zad_shift = 0 # 巡回シフト量

    Ms = [16] # 多値変調数
    code_rates = [1/2, 3/4] # 符号化率

    conv_type = 'soft' # soft/hard
    channel_type = 'Rayleigh' # AWGN/Rayleigh

    last_snr = 5

    size = 19 # DFT行列サイズ
    pilot_len = 9 # パイロット長

    Communication.simulation(flame=flame, user=user, BS=BS, Nc=Nc, CP=CP, path=path, Ps=Ps, BW=BW,
                             data_symbol=data_symbol, Ms=Ms, code_rates=code_rates, conv_type=conv_type,
                             zad_len=zad_len, zad_num=zad_num, zad_shift=zad_shift,
                             channel_type=channel_type, csv_column=[Ms, code_rates], last_snr=last_snr)

if __name__ == '__main__':
    main()