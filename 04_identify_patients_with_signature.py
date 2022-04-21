"""
Trains an autoencoder to perform aberration detection, then uses said model to identify aberrant patients displaying a
signature
"""
import argparse
import os
import pickle
import numpy as np
import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import scipy.stats as st

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('working_dir', type=str)
    args = parser.parse_args()

    # Build dataset for Autoencoder Training
    # - dictionary with case patient id as key, and corresponding neighbor patient ids stored in a list
    with open(os.path.join(args.working_dir, 'top100_neighbors.pkl'), 'rb') as t100:
        neighbors_idx = pickle.load(t100)
    # - dictionary with reference patient id as key, corresponding patient representations as values
    with open(os.path.join(args.working_dir, "reference_patient_emb_withid.pkl"), 'rb') as pwid_pkl:
        patient_reps_ids = pickle.load(pwid_pkl)
    # - Do join
    reference_patient = []
    for neighbor in neighbors_idx.values():
        reference_patient = list(set(reference_patient + neighbor))
    p_reps = []
    for p_id in reference_patient:
        p_reps.append(patient_reps_ids[p_id])
    data = np.array(p_reps)
    with open(os.path.join(args.working_dir, 'reference_pids.pkl'), 'wb') as f1:
        pickle.dump(reference_patient, f1)
    with open(os.path.join(args.working_dir, 'reference_prep.pkl'), 'wb') as f2:
        pickle.dump(data, f2)

    # Now train autoencoder
    with open(os.path.join(args.working_dir, "case_patient_indexes.pkl"), "rb") as file3:
        data_id = pickle.load(file3)
    with open(os.path.join(args.working_dir, "case_patient_emb.pkl"), "rb") as file2:
        case_data = pickle.load(file2)

    # data preprocess
    # train data
    # test data
    # data shape should be (# of data, length of data, 1, 1) type should be float32, range 0-1
    # training data and validation data

    # data = data.reshape((33489, 200, 1))
    print(data.shape)
    data_max = np.max(data)
    data = data / np.max(data)

    # data split
    X, test_X, Y, test_Y = train_test_split(data, data, test_size=0.1, random_state=11)
    train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.1, random_state=11)

    # parameters
    batch_size = 128
    epochs = 10

    input_seqs = Input(shape=(200,))

    latent_dim = 32


    class Autoencoder(Model):
        def __init__(self, latent_dim):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim
            self.encoder = keras.Sequential([
                Dense(latent_dim, activation='relu')
            ])
            self.decoder = keras.Sequential([
                Dense(200, activation='sigmoid')  # Confused....
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


    autoencoder = Autoencoder(latent_dim)

    autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

    autoencoder_train = autoencoder.fit(train_X, train_Y,
                                        epochs=epochs,
                                        shuffle=True,
                                        validation_data=(val_X, val_Y))

    # pass in all the TOI patient representation and see the loss if it pass a certain threshold use predict function
    # pred = autoencoder.predict(test_X)
    # print(pred.shape)

    # draw the loss curve

    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']

    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    case_data = case_data / data_max
    case_pred = autoencoder.predict(case_data)
    i = 0
    error_list = []
    for i in range(1236):
        error_list.append(mean_squared_error(case_data[i], case_pred[i]))

    print(error_list)
    print(np.mean(error_list))
    # print(error_list.sort(reverse=True))
    # print(error_list)
    ci = st.t.interval(alpha=0.95, df=len(error_list) - 1, loc=np.mean(error_list), scale=st.sem(error_list))

    print("this first__________then ci")
    print(ci)
    anomaly_id = []
    for idx, error in enumerate(error_list):
        if error > ci[1]:
            anomaly_id.append(data_id[idx])

    with open(os.path.join(args.working_dir, "aberrant_case_patients.pkl"), "wb") as file4:
        pickle.dump(anomaly_id, file4)






