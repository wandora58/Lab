
import math

from Controller import Communication


def main():
    flame = 2 #シミュレーション回数

    user = 6 # ユーザー数
    user_antenna = 1 # ユーザのアンテナ数

    BS = 1 # 基地局数
    BS_antenna = 4 # 基地局のアンテナ数

    select_user = math.floor(BS_antenna/user_antenna) # 選択されたユーザ数

    Nc = 72 # 搬送波数

    CP = 1 # CP数
    path = 1 # パス数
    Ps = 1.0 # 総送信電力

    data_symbol = 72 # 1フレーム内のシンボル数

    M = 4 # [4,16,64,256]
    conv_type = 'soft' # [soft,hard]
    code_rate = 1/2 # [1/2,2/3,3/4,5/6]
    channel_type = 'Rayleigh' # AWGN/Rayleigh

    last_snr = 5

    zad_len = 4 # Zadoff_Chu系列長　
    zad_num = 1 # Zadoff_Chu系列番号
    zad_shift = 0 # 巡回シフト量

    Communication.simulation(flame=flame, user=user, user_antenna=user_antenna, BS=BS, BS_antenna=BS_antenna, select_user=select_user,
                             Nc=Nc, CP=CP, path=path, Ps=Ps, data_symbol=data_symbol, M=M, conv_type=conv_type, code_rate=code_rate,
                             zad_len=zad_len, zad_num=zad_num, zad_shift=zad_shift,
                             channel_type=channel_type, last_snr=last_snr)

if __name__ == '__main__':
    main()

