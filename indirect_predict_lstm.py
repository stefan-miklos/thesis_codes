import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
import numpy as np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense




#data
k_t= np.asarray([10.3014652804168 , 6.46192412728571 , 7.66457419315423 , 8.43010905303833 , 6.67268044442984 , 4.58052624133982 , 6.49003099578714 , 6.72232655966373 , 5.06454171307806 , 8.59439065500007 , 2.83849422343631 , 5.72739942407955 , 11.8461234224553 , 2.57647123622468 , 10.9701104019758 , 13.658081495753 , 12.68781400135 , 15.0202090324646 , 16.6299226114657 , 17.3953958299271 , 18.348837794066 , 19.0839745826479 , 18.7503341196848 , 19.5819097298401 , 19.4309217443775 , 19.5825528269231 , 19.2117650791035 , 18.6342850418289 , 18.9890006265515 , 18.0416392224268 , 18.2911180151597 , 17.6995996615566 , 17.1982879393888 , 17.0859536369512 , 16.4410160496531 , 15.9979401774788 , 15.4783678711509 , 14.6920033136865 , 14.1591536363998 , 14.1153019294467 , 13.7801330748213 , 13.3669816182764 , 13.3268881173952 , 13.1104693068531 , 12.5917115232613 , 12.1923568918688 , 11.8772018290486 , 11.4612805465146 , 11.3710718370731 , 11.3491223015764 , 10.5963363975756 , 10.2523649769121 , 10.1481470285456 , 10.3017851262142 , 9.95043702511892 , 10.0156588245332 , 9.61892725278353 , 9.5389388023916 , 9.23684094927337 , 9.12745823815448 , 9.15554839567375 , 9.01426648423271 , 8.96929053974414 , 8.76443064452307 , 8.71618188835819 , 9.03991817860984 , 8.64980284954269 , 8.99183961062034 , 8.74595391457023 ,8.92650845104465 , 9.70251581643217])
train = k_t[:k_t.size-5]




def sequence(data, steps):
    X, y = [], []
    for i in range(len(data)):
        end = i + steps
        if end > len(data)-1:
            break
        seq_X, seq_y = data[i:end], data[end]
        X.append(seq_X)
        y.append(seq_y)
    return np.asarray(X), np.asarray(y)

steps = 5
X_train, y_train = sequence(train, steps)

features = 1

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], features))

# building the model
y_hat_list = []

for i in range(30):
    model = Sequential()
    model.add(LSTM(50, activation = "relu",  input_shape = (steps, features)))
    model.add(Dense(1))
    model.compile(optimizer = "adam", loss = "mse")

    model.fit(X_train, y_train, epochs = 200, verbose=0)

    new_input = k_t[-10:-5]
    new_input = new_input.reshape(1, steps, features)

    yhat = model.predict(new_input, verbose = 0)
    y_hat_list.append(yhat)

print(y_hat_list)
sum(y_hat_list)/len(y_hat_list)


