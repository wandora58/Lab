
import math
import numpy as np

from tqdm import tqdm
from keras.utils.np_utils import to_categorical
from keras import backend as K

from Model.GAN import GAN
from Model.utils import Load


def train_generative_adversarial_network(max_epoch, batch_size, sample, test, user, BS, pilot_type, pilot_len, size, input_type):

    load = Load(sample, test, size, pilot_len, user, pilot_type, input_type)
    gan = GAN(max_epoch, batch_size, sample, test, user, BS, pilot_type, pilot_len, size, input_type)

    input_data, input_test, answer_data, answer_test = load.load_data()
    gen_train_stage, dis_train_stage, gen_test_stage = gan.create_model()

    for epoch in tqdm(range(max_epoch)):
        num_batch = int(math.floor(sample / batch_size))
        lr1, lr2 = gan.adjast_learning_rate()

        for batch in range(num_batch):

            now_batch_size = min((batch + 1) * batch_size, sample) - batch * batch_size
            batch_mask = np.random.choice(sample, now_batch_size)

            input_batch = input_data[batch_mask]
            true_batch = answer_data[batch_mask]

            true_label = to_categorical(np.ones(now_batch_size), num_classes=2)
            fake_label = to_categorical(np.zeros(now_batch_size), num_classes=2)

            K.set_value(gen_train_stage.optimizer.lr, lr1)
            gen_score = gen_train_stage.train_on_batch([input_batch, true_batch], [true_batch, true_label])

            K.set_value(gen_train_stage.optimizer.lr, lr2)
            dis_train_stage.train_on_batch([input_batch, true_batch], [fake_label, true_label])

        NMSE = gan.calculate_NMSE(epoch, input_test, answer_test, gen_test_stage)

    gan.illustrate()
    gan.save_model(gen_test_stage)
