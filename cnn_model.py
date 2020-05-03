"""
    File name: train.py
    Function Des:

    ~~~~~~~~~~

    author: Skyduy <cuteuy@gmail.com> <http://skyduy.me>

"""
import os
import numpy as np
from keras import layers
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model
import load_data
from scipy.stats import spearmanr
from scipy.stats import pearsonr

def prepare_data():
    return (np.array(load_data.train_x), np.array(load_data.train_y),
                np.array(load_data.test_x), np.array(load_data.test_y))


def build_model():
    print('... construct network')


    inputs = layers.Input((32, 32, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='linear')(x)

    return Model(inputs=inputs, outputs=out)


def train(pic_folder, weight_folder):
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)
    x_train, y_train, x_test, y_test = prepare_data()
    model = build_model()

    print('... compile models')
    model.compile(
        optimizer='adam', # make the learning rate smaller
        loss='mse', # may be not good
    )

    print('... begin train')

    check_point = ModelCheckpoint(
        os.path.join(weight_folder, '{epoch:02d}.hdf5'))

    class TestAcc(Callback):
        def on_epoch_end(self, epoch, logs=None):
            weight_file = os.path.join(
                weight_folder, '{epoch:02d}.hdf5'.format(epoch=epoch + 1))
            model.load_weights(weight_file)
            print("x shape: ", np.shape(x_test))
            out = model.predict(x_test, verbose=1)

            # test data spearman
            corr, p_value = spearmanr(y_test, out)
            print("test data spearman corealation is: %.4f" % corr)
            print("test data spearman p value is: %.4f" % p_value)

            # test data pearson
            # corr, p_value = pearsonr(y_test, out)
            # print("test data pearson corealation is: %.4f" % corr)
            # print("test data pearson p value is: %.4f" % p_value)

            # Yousong's accuracy
            # predict = np.array([np.argmax(i) for i in out])
            # answer = np.array([np.argmax(i) for i in y_test])
            # # print("y_test: ", y_test)
            # # print("predict: ", predict, "answer: ", answer)
            # acc = np.sum(predict == answer) / len(predict)
            # print('Single phone test accuracy: {:.2%}'.format(acc))
            # print('----------------------------------\n')

    model.fit(
        x_train, y_train, batch_size=128, epochs=5,
        validation_split=0.1, callbacks=[check_point, TestAcc()],
    )


if __name__ == '__main__':
    train(
        pic_folder=r'',
        weight_folder=r'CNN_keras_models'
    )
