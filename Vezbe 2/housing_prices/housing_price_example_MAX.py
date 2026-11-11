import os

# Mora biti pre importa Keras-a i biblioteka koje koriste računarske niti.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Ova varijanta koristi CPU.
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json
import math
import multiprocessing
import shutil
import tempfile
import time

import pandas
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

from custom_plotting import CustomPlotting
from custom_preparer import CustomPreparer
from scorer import Scorer


BEST_MODEL_PATH = 'best_model.keras'
BEST_PARAMETERS_PATH = 'best_model_parameters.json'

NUMBER_OF_COLUMNS = 14
SHARE_FOR_TRAINING = 0.8
RANDOM_SEED = 42

# Broj logičkih procesora; Windows ProcessPoolExecutor podržava najviše 61 radnika.
MAX_PROCESSES = min(os.cpu_count() or 1, 61) if os.name == 'nt' else (os.cpu_count() or 1)


def train_combination(index, parameters, trainX, trainY, testX, preparer, temp_directory):
    import keras
    from ann_regression import AnnRegression

    keras.backend.clear_session()

    time_begin = time.perf_counter()

    ann = AnnRegression(**parameters, verbose=0)
    trainPredict, testPredict = ann.compile_fit_predict(trainX, trainY, testX)

    trainPredict, trainYOriginal, testPredict, testYOriginal = (
        preparer.inverse_transform(trainPredict, testPredict)
    )

    train_score, test_score = Scorer().get_score(
        trainYOriginal, trainPredict,
        testYOriginal, testPredict,
    )

    # Svaki zadatak snima svoj fajl, pa nema prepisivanja između procesa.
    model_path = os.path.join(temp_directory, f'model_{index}.keras')
    ann.model.save(model_path)

    duration = time.perf_counter() - time_begin

    keras.backend.clear_session()

    return parameters, train_score, test_score, duration, model_path


def main():
    from ann_regression import AnnRegression

    dataframe = pandas.read_csv('housing.csv', engine='python', sep=';', header=None)
    dataframe = dataframe.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    preparer = CustomPreparer(dataframe, NUMBER_OF_COLUMNS, SHARE_FOR_TRAINING)
    trainX, trainY, testX, testY = preparer.prepare_for_training()

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

    combinations = []

    for layers in architectures:
        layer_activations = activations if layers else ['relu']

        for activation, loss, batch_size in product(
            layer_activations, losses, batch_sizes
        ):
            combinations.append({
                'hidden_layers': layers,
                'activation_function': activation,
                'kernel_initializer': (
                    'he_normal' if activation == 'relu' else 'glorot_uniform'
                ),
                'cost_function': loss,
                'batch_size_number': batch_size,
            })

    best_test_score = float('inf')
    best_parameters = None
    workers = min(MAX_PROCESSES, len(combinations))

    print(f'Kombinacija: {len(combinations)} | Procesa: {workers}')

    # Privremeni modeli brišu se po završetku.
    with tempfile.TemporaryDirectory() as temp_directory:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),) as executor:
            futures = [executor.submit(
                    train_combination,
                    index,
                    parameters,
                    trainX,
                    trainY,
                    testX,
                    preparer,
                    temp_directory,
                ) for index, parameters in enumerate(combinations)
            ]

            for completed, future in enumerate(as_completed(futures), start=1):
                parameters, train_score, test_score, duration, model_path = (
                    future.result()
                )

                text = (
                    f'Završeno: {completed}/{len(combinations)}\n'
                    f'Parametri: {parameters}\n'
                    f'Time: {duration:.2f} seconds\n'
                    f'Train Score: {train_score:.3f} RMSE\n'
                    f'Test Score: {test_score:.3f} RMSE\n\n'
                )

                print(text, end='')

                # Samo glavni proces upisuje rezultate i najbolji model.
                with open('results.txt', 'a', encoding='utf-8') as file:
                    file.write(text)

                if math.isfinite(test_score) and test_score < best_test_score:
                    best_test_score = float(test_score)
                    best_parameters = {
                        **parameters,
                        'test_rmse': best_test_score,
                    }

                    shutil.copyfile(model_path, BEST_MODEL_PATH)

                    with open(
                        BEST_PARAMETERS_PATH, 'w', encoding='utf-8'
                    ) as file:
                        json.dump(best_parameters, file, indent=2)

                    print(f'Novi najbolji RMSE: {best_test_score:.3f}')

                os.remove(model_path)

    if best_parameters is None:
        raise RuntimeError('Nijedan model nije dao konačan RMSE.')

    print(f'Najbolji test RMSE: {best_test_score:.3f}')
    print(f'Najbolji parametri: {best_parameters}')

    ann = AnnRegression()
    ann.use_current_model(BEST_MODEL_PATH, trainX)
    trainPredict, testPredict = ann.get_predict(testX)

    trainPredict, trainYOriginal, testPredict, testYOriginal = (
        preparer.inverse_transform(trainPredict, testPredict)
    )

    CustomPlotting().show_plots(testPredict, testYOriginal)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()