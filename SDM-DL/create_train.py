
from Model.Train import Train

def main():

    flame = 10500
    symbol = 128
    user = 19
    BS = 1

    CP = 1
    path = 1

    Ps = 1.0
    EbN0 = 20

    pilot_len = 19
    size = 19
    type = 'DFT'

    for pilot_len in range(19,8,-1):
        Train(flame=flame, symbol=symbol, user=user, BS=BS, CP=CP, path=path,
              Ps=Ps, EbN0=EbN0, pilot_len=pilot_len, size=size, type=type).create_train()



if __name__ == '__main__':
    main()