# Vezbe iz predmeta Inteligentni softverski infrastrukturni sistemi
# Koriscenje neuronske mreze na primeru predvidjanja cena kuca
#
# Skup podataka: Boston Housing (506 kuća, 13 atributa + cijena).
# Kolone (po redu, vidi housing.txt): CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
# Cilj: na osnovu 13 atributa predvidjeti MEDV = medijana cijene kuće u hiljadama dolara.
#
# Tok programa:  učitaj CSV -> izmiješaj redove -> podijeli + skaliraj (CustomPreparer)
#                -> treniraj + predvidi (AnnRegression) -> vrati u originalne jedinice
#                -> izračunaj RMSE (Scorer) -> nacrtaj (CustomPlotting)

import time
import pandas
import json
from ann_regression import AnnRegression
from custom_plotting import CustomPlotting
from custom_preparer import CustomPreparer
from scorer import Scorer
from itertools import product
from concurrent.futures import ProcessPoolExecutor


BEST_MODEL_PATH = 'best_model.keras'
BEST_PARAMETERS_PATH = 'best_model_parameters.json'

NUMBER_OF_COLUMNS = 14      # 13 ulaznih atributa + 1 ciljna kolona (cijena)
SHARE_FOR_TRAINING = 0.85   # 85% podataka za trening, 15% za test
RANDOM_SEED = 42            # fiksni seed da miješanje redova bude isto pri svakom pokretanju

# load the dataset
# sep=';' jer je CSV odvojen tačka-zarezom; header=None jer fajl nema red sa nazivima kolona
dataframe = pandas.read_csv('housing.csv', engine='python', sep=';', header=None)

# shuffle rows
# CSV nije nasumično poređan (posljednje kuće su jeftinije), pa bi bez miješanja
# test skup imao drugačiju raspodjelu od trening skupa i ocjena modela ne bi bila realna.
dataframe = dataframe.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# prepare data
preparer = CustomPreparer(dataframe, NUMBER_OF_COLUMNS, SHARE_FOR_TRAINING)
trainX, trainY, testX, testY = preparer.prepare_for_training()   # svi u skaliranom opsegu [0, 1]


# make predictions
best_test_score = float('inf')
best_parameters = None

architectures = [
    (),
    (8,),
    (16,),
    (32,),
    (64,),
    (128,),
    (16, 8),
    (32, 16),
    (64, 32),
    (128, 64),
    (32, 16, 8),
    (64, 32, 16),
]

activations = ['relu', 'tanh', 'sigmoid']
losses = ['mean_squared_error', 'mean_absolute_error']
batch_sizes = [8, 16, 32, 64]

for layers in architectures:
    # Bez skrivenih slojeva aktivacija skrivenih slojeva nema uticaja.
    layer_activations = activations if layers else ['relu']

    for activation, loss, batch_size in product(
        layer_activations, losses, batch_sizes
    ):
        
        ann = AnnRegression(
            hidden_layers=layers,
            activation_function=activation,
            kernel_initializer='he_normal' if activation == 'relu' else 'glorot_uniform',
            cost_function=loss,
            batch_size_number=batch_size,
            verbose=0
        )

        print(f"Cost function: {loss}, Activation function: {activation}, Hidden layers: {layers}, Batch size: {batch_size}")
        
        time_begin = time.time()
        trainPredict, testPredict = ann.compile_fit_predict(trainX, trainY, testX)
        time_end = time.time()
        print('Training duration: ' + str(time_end - time_begin) + ' seconds')

        # invert predictions
        # predikcije su u opsegu [0, 1]; ovdje se vraćaju u hiljade dolara da bi RMSE imao smisla
        trainPredictOriginal, trainYOriginal, testPredictOriginal, testYOriginal = preparer.inverse_transform(trainPredict, testPredict)

        # calculate root mean squared error
        scorer = Scorer()
        trainScore, testScore = scorer.get_score(trainYOriginal, trainPredictOriginal, testYOriginal, testPredictOriginal)
        print('Train Score: %.2f RMSE' % trainScore)
        print('Test Score: %.2f RMSE' % testScore)

        with open('results.txt', 'a', encoding='utf-8') as results_file:
            results_file.write(f"Cost function: {loss}, Activation function: {activation}, Hidden layers: {layers}, Batch size: {batch_size}\n")
            results_file.write(f"Time: {time_end - time_begin} seconds\n")
            results_file.write(f"Train Score: {trainScore} RMSE\n")
            results_file.write(f"Test Score: {testScore} RMSE\n\n")

        if testScore < best_test_score:
            best_test_score = testScore
            best_parameters = {
                'cost_function': loss,
                'activation_function': activation,
                'hidden_layers': layers,
                'batch_size_number': batch_size,
                'test_rmse': testScore,
            }
            ann.model.save(BEST_MODEL_PATH)
            with open(BEST_PARAMETERS_PATH, 'w', encoding='utf-8') as parameters_file:
                json.dump(best_parameters, parameters_file, indent=2)
            print(f'Novi najbolji model je sačuvan: {BEST_MODEL_PATH}')


print(f'Najbolji test RMSE: {best_test_score:.2f}')
print(f'Parametri su sačuvani u: {BEST_PARAMETERS_PATH}')

# Učitavanje najboljeg modela pronađenog u svim iteracijama
ann_regression = AnnRegression()
ann_regression.use_current_model(BEST_MODEL_PATH, trainX)
trainPredict, testPredict = ann_regression.get_predict(testX)

# Vraćanje predikcija u originalne jedinice (hiljade dolara)
trainPredict, trainYOriginal, testPredict, testYOriginal = preparer.inverse_transform(trainPredict, testPredict)

# Prikaz konačnog rezultata najboljeg modela
scorer = Scorer()
trainScore, testScore = scorer.get_score(trainYOriginal, trainPredict, testYOriginal, testPredict)
print('Best train Score: %.2f RMSE' % trainScore)
print('Best test Score: %.2f RMSE' % testScore)

# plotting
# plava linija = predikcija, narandžasta = stvarna cijena (samo test skup); plt.show() čeka da se prozor zatvori
custom_plotting = CustomPlotting()
custom_plotting.show_plots(testPredict, testYOriginal)
