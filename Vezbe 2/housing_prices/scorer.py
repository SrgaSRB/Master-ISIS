# Ocjena modela: RMSE (Root Mean Squared Error) = korijen srednje kvadratne greške.
# Izražen je u istim jedinicama kao cilj (hiljade dolara), pa je lako interpretirati:
# RMSE 5 znači da model u prosjeku griješi ~5000$.

import math
from sklearn.metrics import mean_squared_error


class Scorer:

    def get_score(self, trainY, trainPredict, testY, testPredict):
        trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))   # greška na trening skupu
        testScore = math.sqrt(mean_squared_error(testY, testPredict))      # greška na test skupu (važnija!)
        # Ako je trainScore mnogo manji od testScore, model je "prenaučen" (overfitting).
        return trainScore, testScore
