# Priprema podataka za mrežu: podjela na trening/test, skaliranje, razdvajanje ulaza (X) i cilja (Y)
# i vraćanje predikcija u originalni opseg (inverse_transform).

import numpy
from sklearn.preprocessing import MinMaxScaler


class CustomPreparer:

    def __init__(self, dataframe, number_of_columns, share_for_training):
        # MinMaxScaler skalira SVAKU kolonu posebno u opseg [0, 1]:  x' = (x - min) / (max - min)
        # Neuronske mreže (naročito sa sigmoidom) mnogo bolje rade kad su ulazi istog reda veličine.
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.datasetOrig = dataframe.values                      # pandas DataFrame -> numpy matrica (506 x 14)
        self.datasetOrig = self.datasetOrig.astype('float32')    # float32 je standardni tip za Keras/TensorFlow
        self.number_of_columns = number_of_columns               # 14 = 13 ulaznih atributa + 1 ciljna kolona
        self.predictor_column_no = self.number_of_columns - 1    # indeks ciljne kolone (posljednja = 13 = MEDV, cijena kuće)
        self.share_for_training = share_for_training             # udio podataka za trening (0.85 = 85%)

    def prepare_for_training(self):
        """Dijeli podatke na trening/test, skalira ih i vraća (trainX, trainY, testX, testY)."""
        # Podjela: prvih 85% redova = trening, ostatak = test.
        # (Redovi su već nasumično izmiješani u housing_price_example.py.)
        train_size = int(len(self.datasetOrig) * self.share_for_training)
        train = self.datasetOrig[0:train_size, :]
        test = self.datasetOrig[train_size:, :]
        print(len(train), len(test))

        # Scaler uči min/max SAMO na trening skupu (fit_transform), a test skup se samo
        # transformiše tim istim parametrima - tako model ne "vidi" ništa iz test skupa.
        train = self.scaler.fit_transform(train)
        test = self.scaler.transform(test)

        # Razdvajanje na ulaze X (13 kolona) i cilj Y (1 kolona)
        trainX, trainY = self.create_dataset(train)
        testX, testY = self.create_dataset(test)

        # Čuvaju se u objektu jer su potrebni u inverse_transform.
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        return trainX.copy(), trainY.copy(), testX.copy(), testY.copy()

    def inverse_transform(self, trainPredict, testPredict):
        """Vraća predikcije i stvarne vrijednosti iz opsega [0,1] u originalne jedinice (hiljade dolara).

        Trik: MinMaxScaler zna da inverzno skalira samo CIJELE redove sa 14 kolona.
        Zato se predikcija (1 kolona) "zalijepi" na ulaze X (13 kolona) da bi se dobio red od 14 kolona,
        uradi inverse_transform, pa se iz rezultata izvuče samo posljednja kolona.
        """
        # Sve svodimo na oblik (n, 1) da bi concatenate radio
        trainPredict = numpy.reshape(trainPredict, (-1, 1))
        testPredict = numpy.reshape(testPredict, (-1, 1))
        trainY = numpy.reshape(self.trainY, (-1, 1))
        testY = numpy.reshape(self.testY, (-1, 1))

        # X + predikcija  -> (n, 14)   i   X + stvarni Y -> (n, 14)
        trainXAndPredict = numpy.concatenate((self.trainX, trainPredict), axis=1)
        testXAndPredict = numpy.concatenate((self.testX, testPredict), axis=1)
        trainXAndY = numpy.concatenate((self.trainX, trainY), axis=1)
        testXAndY = numpy.concatenate((self.testX, testY), axis=1)

        # Inverzno skaliranje svih kolona nazad u originalni opseg
        trainXAndPredict = self.scaler.inverse_transform(trainXAndPredict)
        trainXAndY = self.scaler.inverse_transform(trainXAndY)
        testXAndPredict = self.scaler.inverse_transform(testXAndPredict)
        testXAndY = self.scaler.inverse_transform(testXAndY)

        # Uzima se samo ciljna kolona (indeks 13)
        trainPredict = trainXAndPredict[:, self.predictor_column_no]
        trainY = trainXAndY[:, self.predictor_column_no]
        testPredict = testXAndPredict[:, self.predictor_column_no]
        testY = testXAndY[:, self.predictor_column_no]
        return trainPredict, trainY, testPredict, testY

    def create_dataset(self, dataset):
        """Razdvaja matricu na ulaze X (prvih 13 kolona) i cilj Y (posljednja, 14. kolona)."""
        dataX = dataset[:, 0:self.predictor_column_no]   # kolone 0..12 = ulazni atributi
        dataY = dataset[:, self.predictor_column_no]     # kolona 13 = cijena (MEDV)
        return dataX, dataY
