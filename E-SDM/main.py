
from Controller import Communication


def main():
    flame = 2 #シミュレーション回数

    user = 4 # ユーザー数
    BS = 4 # 基地局数
    Nc = 72 # 搬送波数

    CP = 1 # CP数
    path = 1 # パス数
    Ps = 1.0 # 総送信電力

    BW = 1/6 # チャネル帯域幅[Mhz]
    data_symbol = 72 # 1フレーム内のシンボル数

    total_bit = user * 2 # 全ストリーム合計のシンボルあたりのビット数

    conv_type = 'soft' # soft/hard
    channel_type = 'Rayleigh' # AWGN/Rayleigh

    last_snr = 5

    zad_len = 4 # Zadoff_Chu系列長　
    zad_num = 1 # Zadoff_Chu系列番号
    zad_shift = 0 # 巡回シフト量

    Communication.simulation(flame=flame, user=user, BS=BS, Nc=Nc, CP=CP, path=path, Ps=Ps, BW=BW,
                             data_symbol=data_symbol, conv_type=conv_type, total_bit=total_bit,
                             zad_len=zad_len, zad_num=zad_num, zad_shift=zad_shift,
                             channel_type=channel_type, last_snr=last_snr)

if __name__ == '__main__':
    main()

