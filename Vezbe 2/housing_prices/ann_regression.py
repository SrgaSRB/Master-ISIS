# Neuronska mreža za REGRESIJU (predviđanje kontinualne vrijednosti - cijene kuće).
# Nasljeđuje hiperparametre iz AnnBase i na osnovu njih gradi Keras Sequential model.

from keras import Input                 # Input = eksplicitni ulazni sloj (preporučeni način u Keras 3)
from keras.layers import Dense, Dropout # Dense = potpuno povezani (fully connected) sloj
from keras.models import Sequential     # Sequential = model gdje se slojevi ređaju jedan za drugim
from keras.models import load_model     # učitavanje snimljenog modela sa diska
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

from ann_base import AnnBase

MODEL_NAME = 'current_model.keras'   # putanja pod kojom se model snima/učitava (Keras 3 zahtijeva ekstenziju .keras)


class AnnRegression(AnnBase):

    def get_model(self, number_of_inputs):
        """Gradi arhitekturu mreže na osnovu hiperparametara iz AnnBase.
        number_of_inputs = broj ulaznih atributa (kolona u X); čita se iz podataka,
        pa model automatski prati promjenu broja kolona u CSV-u.
        """

        model = Sequential()

        # Ulazni sloj: samo definiše oblik jednog uzorka - vektor od number_of_inputs brojeva
        model.add(Input(shape=(number_of_inputs,)))

        for neurons in self.hidden_layers:
            model.add(Dense(
                neurons,
                activation=self.activation_function,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=l2(self.l2_strength),
            ))

            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))

        model.add(Dense(1, activation='linear'))

        return model

    def get_model_from_path(self, path):
        """Učitava već istreniran i snimljen model sa diska."""
        return load_model(path)

    def compile_and_fit(self, trainX, trainY):
        """Kompajlira model (zadaje loss i optimizator) i trenira ga na trening podacima."""
        self.model = self.get_model(trainX.shape[1])   # trainX.shape[1] = broj ulaznih atributa (13)
        # compile: povezuje model sa funkcijom greške i optimizatorom
        self.model.compile(loss=self.cost_function, optimizer=self.optimizer)
        self.trainX = trainX   # čuva se da bi get_predict mogao da izračuna i predikciju na trening skupu
        # fit: samo treniranje - epochs i batch_size dolaze iz AnnBase
        self.model.fit(trainX, trainY, epochs=self.epoch_number, batch_size=self.batch_size_number, verbose=self.verbose)
        #self.model.save(MODEL_NAME)   # snimanje modela na disk (isključeno)

    def use_current_model(self, path, trainX):
        """Alternativa treniranju: umjesto fit-a, učitaj ranije snimljen model."""
        self.trainX = trainX
        self.model = self.get_model_from_path(path)

    def get_predict(self, testX):
        """Vraća predikcije i za trening i za test skup (obje u skaliranom 0..1 opsegu), oblika (n, 1)."""
        trainPredict = self.model.predict(self.trainX)   # koliko dobro mreža "pamti" ono na čemu je trenirana
        testPredict = self.model.predict(testX)          # koliko dobro generalizuje na neviđene podatke
        return trainPredict, testPredict

    def compile_fit_predict(self, trainX, trainY, testX):
        """Glavna metoda koju poziva housing_price_example.py: treniraj pa predvidi."""
        self.compile_and_fit(trainX, trainY)
        #self.use_current_model(MODEL_NAME, trainX)   # alternativa: otkomentarisati ovo, a zakomentarisati liniju iznad
        return self.get_predict(testX)
