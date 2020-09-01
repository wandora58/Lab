
from Controller import train_gan
from Controller import train_lightgbm

def main():
    max_epoch = 100  # エポック数
    batch_size = 100  # バッチサイズ

    sample = 10000  # サンプル数
    test = 500  # テスト数

    user = 20  # ユーザー数
    BS = 20  # 基地局数

    total_bit = user*2

    SNR = 20  # int/random
    input_type = 'eigen'  # eigen/channel

    num_boost_round = 50
    early_stopping_round = 100

    # train_gan.train_generative_adversarial_network(max_epoch=max_epoch, batch_size=batch_size,
    #                                                sample=sample, test=test, user=user, BS=BS,
    #                                                total_bit=total_bit, SNR=SNR, input_type=input_type)

    train_lightgbm.train_lightgbm(sample=sample, test=test, user=user,
                                  total_bit=total_bit, SNR=SNR, input_type=input_type,
                                  num_boost_round=num_boost_round, early_stopping_round=early_stopping_round)



if __name__ == '__main__':
    main()