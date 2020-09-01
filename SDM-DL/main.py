
from Controller import train_gan


def main():
    max_epoch = 100  # エポック数
    batch_size = 100  # バッチサイズ

    sample = 10000  # サンプル数
    test = 500  # テスト数

    user = 19  # ユーザー数
    BS = 1  # 基地局数

    pilot_type = 'DFT' # frame / DFT
    pilot_len = 9 # パイロット長
    size = 19 # DFTサイズ

    input_type = 'ls' # ls / receive

    pil = ['DFT', 'frame']
    inp = ['ls', 'receive']

    # train_gan.train_generative_adversarial_network(max_epoch=max_epoch, batch_size=batch_size,
    #                                                sample=sample, test=test, user=user, BS=BS,
    #                                                pilot_type=pilot_type, pilot_len=pilot_len, size=size,
    #                                                input_type=input_type)

    for p in pil:
        for i in inp:
            train_gan.train_generative_adversarial_network(max_epoch=max_epoch, batch_size=batch_size,
                                                           sample=sample, test=test, user=user, BS=BS,
                                                           pilot_type=p, pilot_len=pilot_len, size=size,
                                                           input_type=i)


if __name__ == '__main__':
    main()