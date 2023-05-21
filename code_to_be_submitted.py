# a szüksges csomagok importálása
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# a visszatekintéses bemeneti mátrix és kimeneti vektor létrehozására szolgáló függvény
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

#a prediktáló vektor utolsó elemét a predikcióval helyettesítő függvény
def shift_vector(vector, new_vector):
    for index,country in enumerate(vector):
        res = np.empty_like(country)
        res[0:2] = country[1:]
        one_hot_code = country[0][1:]
        res[2] = np.concatenate((new_vector[index], one_hot_code))
        vector[index] = res
    return vector

# az országonkénti bemeneti részmátrixot létrehozó függvény
def make_matrix(ts, steps, x):
    X, y = sequence(ts, steps)
    X_res =  np.empty(shape=(len(X),steps,6))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            lis = np.zeros(shape=(6,))
            lis[0] = X[i][j]
            lis[x] = 1
            X_res[i][j] = lis
    return X_res, y

# Az országok k_t idősorai
k_t_aus = np.asarray([52.1665646240446 , 45.2253112722121 , 49.6212045219086 , 48.8529869261414 , 44.0884127697363 , 48.673734557087 , 43.8985888509474 , 47.0236736425682 , 46.4628144573837 , 48.4920584345725 , 47.7469060851776 , 45.170979768944 , 41.656464629472 , 37.2298317571504 , 37.6057193570369 , 38.6131663078275 , 36.672000569734 , 32.1398240378826 , 33.5637742885928 , 29.1003010918104 , 28.1873576439313 , 27.1041454619755 , 24.2749215131762 , 25.4461945014997 , 18.4208221761529 , 18.6635336316244 , 14.0932945695679 , 9.692949112938 , 5.69254992984781 , 4.18779205356227 , 1.84753070903554 , 0.860736725427983 , -1.09632283892541 , -3.60745801798164 , -7.87790157082626 , -8.74331808405922 , -10.7201328949049 , -14.321721169912 , -17.6350417443876 , -19.422582358325 , -23.4282691663772 , -28.9836365042146 , -28.8460832516009 , -28.9132117948715 , -36.1411059282191 , -37.385256523297 , -41.6456344357289 , -44.031361515792 , -46.1876818669787 , -45.1998334542452 , -48.6708332233962 , -53.0958806456038 , -51.2402959310066 , -54.032040471487 , -59.376331337666 , -54.3049080746907 , -61.288672061516 , -59.6587888690771 , -60.9247492468854 , -64.4816558179814])
k_t_hun = np.asarray([22.8909097700725 , 12.8002890972752 , 26.5435504592453 , 12.7876403107331 , 10.6606793123222 , 17.7429248004767 , 6.18115488694618 , 13.2063636814778 , 17.6896676276429 , 16.898282726236 , 18.6810335249665 , 19.5203641175888 , 12.4291720421073 , 14.8972523466136 , 14.8122243373346 , 17.6641175501915 , 17.0632620796061 , 15.0834559179495 , 21.699556227865 , 17.2466988514457 , 24.3207588912231 , 23.1899888922337 , 22.1387707458475 , 25.6309657473949 , 23.3376259872165 , 23.6071762294088 , 22.4714944416603 , 17.5548851811746 , 13.9548789877594 , 17.6349665993443 , 17.6214072521248 , 15.5560323264708 , 18.2730058874677 , 18.7988501641539 , 14.5113520487768 , 11.8642889769069 , 8.46576449581512 , 3.41449938964122 , 3.4235430644048 , 4.69303238984647 , -4.96831705287341 , -10.9470048627619 , -12.0155995393824 , -10.2744043228021 , -15.9856896386593 , -14.160201634804 , -21.1130893327311 , -21.7330854603076 , -27.5316695396099 , -29.4624466138913 , -31.739088718147 , -36.5228857008017 , -38.5128556710842 , -44.1699396018897 , -47.1944846395737 , -42.3435030323752 , -50.3990940690082 , -46.3650662217665 , -49.0095175929145 , -52.7374773662942])
k_t_slo = np.asarray([17.1151159466903 , 13.0516326277603 , 21.4615972714993 , 13.8992116579673 , 10.1964957076254 , 18.0053808773655 , 17.0437371985609 , 11.3334024442498 , 17.3621160884446 , 22.5338295624356 , 24.5435543176604 , 23.6515919817938 , 16.7597536289277 , 20.3143382461711 , 20.5996502838778 , 18.9422192143459 , 16.7942358571419 , 18.9159732416795 , 18.1086612340651 , 16.0274297505807 , 21.0179090856119 , 17.2387942869496 , 17.377199419749 , 20.7127757063696 , 17.7464133397708 , 18.1103400029165 , 18.6432245832009 , 14.608027975063 , 14.3442598097439 , 16.4529722496024 , 17.071226965028 , 15.8825878118524 , 11.8501572642427 , 8.68074639290383 , 3.92370158570086 , 5.76884074647858 , 0.720609284393993 , 1.30947252796794 , 2.25092420582888 , -1.3450979937815 , -2.28924388328286 , -6.16014599788962 , -9.60226894251831 , -10.28275082887 , -13.9555585242722 , -12.2762885400868 , -15.2899172381406 , -16.6065258419778 , -21.3887007329399 , -25.3151425342347 , -27.0406446381625 , -34.2819108505893 , -36.0121946127794 , -40.2178590537825 , -45.8837425381543 , -42.3473828753286 , -49.8707316543523 , -48.7781453851834 , -51.1261760135981 , -57.730140610206])
k_t_pol = np.asarray([28.2390509667771 , 27.6242276465723 , 30.0415954405875 , 24.0434002339489 , 23.8602450047089 , 19.7128032180042 , 16.9609444739723 , 20.4534091101638 , 15.8955964193162 , 20.7856243972701 , 20.4432138657566 , 24.0028272151402 , 13.6704473686706 , 15.177659663287 , 11.9995998972931 , 16.3096950545236 , 16.0646899144233 , 16.3740769885926 , 18.08526210526 , 14.8018289334918 , 20.8218596799455 , 12.2464168347251 , 11.7056145205063 , 13.9056365488305 , 16.6611354628863 , 20.03050089199 , 17.7597159791795 , 17.2112158323526 , 13.6846274156737 , 15.2490317689974 , 15.8684457657432 , 19.099360852187 , 14.9709516200321 , 13.1737654130889 , 10.2787902958032 , 8.97532348011047 , 7.58986379549667 , 4.36890219665582 , 0.999011898562846 , 1.1017356170259 , -5.66996790328455 , -10.2633028649317 , -14.3629046033212 , -15.0400251320795 , -18.5433669940916 , -19.9053357337087 , -22.5426143147055 , -23.1790973197996 , -25.675742339653 , -26.9430395899901 , -32.5525587871888 , -37.0369432045802 , -37.2799310348261 , -39.7154182660877 , -47.2420872016285 , -44.0485975755518 , -49.8997444763656 , -47.838193509035 , -46.828363402323 , -51.2527470602407])
k_t_cze = np.asarray([22.42877090835 , 21.3228249550439 , 29.7112552494398 , 22.9832841579417 , 22.4241270181411 , 23.4070208676582 , 22.0247849211641 , 23.2630299668843 , 27.6547181410455 , 31.0812726820473 , 32.0840107458687 , 29.6511097255738 , 25.1000790722634 , 28.0047811406867 , 28.3789158007214 , 25.0052815232219 , 24.555150258648 , 24.2592888929842 , 23.8849868763971 , 23.4462189730796 , 29.2931396511703 , 24.5903176660393 , 24.2527087003685 , 26.844013299507 , 24.4593872140505 , 23.4531366669107 , 23.8926124871881 , 18.6763744189971 , 16.5066328551631 , 17.6815320263394 , 18.4795452732492 , 13.4696814772444 , 8.90511233304085 , 5.71904041568446 , 3.93841281152259 , 3.43810575296899 , -2.84326520977511 , -4.2492618237112 , -9.24025095784982 , -10.4405377349401 , -12.883030801427 , -16.1574690434466 , -17.5921657063101 , -16.0618480209083 , -22.6772035224435 , -24.1641358357969 , -30.7882466667179 , -33.5041792074523 , -36.202890556561 , -36.3447262932787 , -39.9781643550282 , -43.0270347196314 , -44.421218106773 , -46.1228084987235 , -53.2844503250754 , -49.5998468025936 , -56.3712107417449 , -54.8213022582289 , -55.7009231671227 , -58.952587274646])

#az országokhoz tartozó részmátrixok előállítása
X_aus,y_aus  = make_matrix(k_t_aus, steps, 1)
X_cze, y_cze = make_matrix(k_t_cze, steps, 2)
X_hun, y_hun = make_matrix(k_t_hun, steps, 3)
X_pol, y_pol = make_matrix(k_t_pol, steps, 4)
X_slo, y_slo = make_matrix(k_t_slo, steps, 5)

#a bemeneti mátrix létrehozása
input_matrix = []
for i in range(X_aus.shape[0]):
    input_matrix.append(X_aus[i]) 
    input_matrix.append(X_cze[i])
    input_matrix.append(X_hun[i])
    input_matrix.append(X_pol[i])
    input_matrix.append(X_slo[i])
input_matrix = np.array(input_matrix)

#a tanuló és tesztelő halmaz szétválasztása
input_matrix_for_learning = input_matrix[:input_matrix.shape[0]-45]
test_matrix = input_matrix[input_matrix.shape[0]-45:]

#az EarlyStopping paramétereinek beállítása
callback = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 50)

#a modell létrehozása a megfelelő paraméterekkel
model = Sequential()
model.add(LSTM(3,
            activation = "relu",
            input_shape =(input_matrix.shape[1], input_matrix.shape[2])))
model.add(Dense(1))    
model.compile(optimizer = "adam", loss = "mse")

history = model.fit(input_matrix_for_learning,
        input_vector_for_learning,
        epochs = 1800,
        batch_size = 100,
        verbose = 0,
        callbacks = [callback], 
        validation_split = 1/10,
        shuffle = False)

#az előrejelzések előkészítése évenként lépve
base_vector = input_matrix_for_learning[-5:]
lstm_2010_pred = model.predict(base_vector, verbose = 0)
vector_upto_2010 = shift_vector(base_vector, lstm_2010_pred)
lstm_2011_pred = model.predict(vector_upto_2010, verbose = 0)
vector_upto_2011 = shift_vector(vector_upto_2010, lstm_2011_pred)
lstm_2012_pred = model.predict(vector_upto_2011, verbose = 0)
vector_upto_2012 = shift_vector(vector_upto_2011, lstm_2012_pred)
lstm_2013_pred = model.predict(vector_upto_2012, verbose = 0)
vector_upto_2013 = shift_vector(vector_upto_2012, lstm_2013_pred)
lstm_2014_pred = model.predict(vector_upto_2013, verbose = 0)
vector_upto_2014 = shift_vector(vector_upto_2013, lstm_2014_pred)
lstm_2015_pred = model.predict(vector_upto_2014, verbose = 0)
vector_upto_2015 = shift_vector(vector_upto_2014, lstm_2014_pred)
lstm_2016_pred = model.predict(vector_upto_2015, verbose = 0)
vector_upto_2016 = shift_vector(vector_upto_2015, lstm_2016_pred)
lstm_2017_pred = model.predict(vector_upto_2016, verbose = 0)
vector_upto_2017 = shift_vector(vector_upto_2016, lstm_2017_pred)
lstm_2018_pred = model.predict(vector_upto_2017, verbose = 0)
vector_upto_2018 = shift_vector(vector_upto_2017, lstm_2018_pred)
lstm_2019_pred = model.predict(vector_upto_2018, verbose = 0)
vector_upto_2019 = shift_vector(vector_upto_2018, lstm_2019_pred)

#az előrejelzések felbontása országonként 
k_t_aus_pred = []
k_t_cze_pred = []
k_t_hun_pred = []
k_t_pol_pred = []
k_t_slo_pred = []
preds = [lstm_2010_pred, lstm_2011_pred, lstm_2012_pred, lstm_2013_pred, lstm_2014_pred,lstm_2015_pred, lstm_2016_pred, lstm_2017_pred, lstm_2018_pred, lstm_2019_pred]
for i in range(len(preds)):
    k_t_aus_pred.append(float(preds[i][0]))
    k_t_cze_pred.append(float(preds[i][1]))
    k_t_hun_pred.append(float(preds[i][2]))
    k_t_pol_pred.append(float(preds[i][3]))
    k_t_slo_pred.append(float(preds[i][4]))
