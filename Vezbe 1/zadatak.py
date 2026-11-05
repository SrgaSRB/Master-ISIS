import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import seaborn as sns
import matplotlib.pyplot as plt


def mae(targets, predictions):
    return np.mean(np.abs(targets - predictions))

def mse(targets, predictions):
    return np.mean(np.square(targets - predictions))

def rmse(targets, predictions):
    return np.sqrt(mse(targets, predictions))

def r2(targets, predictions):
    ss_res = np.sum(np.square(targets - predictions))
    ss_tot = np.sum(np.square(targets - np.mean(targets)))
    return 1 - (ss_res / ss_tot)


df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

df.info()
df.describe()

target = "Zone 1 Power Consumption"

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

df.corr(numeric_only=True)[target].sort_values(ascending=False)

inputs = ["Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows"]

# 80% podataka za razvoj modela, 20% za konačno testiranje
X_dev, X_test, y_dev, y_test = train_test_split(
    df[inputs], df[target], test_size=0.2, random_state=42
)

# Od preostalih 80% izdvajamo 25% za validaciju
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=42
)

def fitness_func(ga_instance, solution, solution_idx):
    selected_columns = [column for column, gene in zip(inputs, solution) if gene == 1]

    if not selected_columns:
        return 0.0

    candidate_model = LinearRegression()
    candidate_model.fit(X_train[selected_columns], y_train)
    predictions = candidate_model.predict(X_val[selected_columns])

    return 1.0 / (1.0 + rmse(y_val, predictions))


ga = pygad.GA(
    num_generations=30,                 # Broj generacija
    sol_per_pop=20,                     # Broj kombinacija kolona u svakoj generaciji
    num_parents_mating=4,               # Broj roditelja koji stvaraju potomke
    num_genes=len(inputs),              # Jedan gen za svaku ulaznu kolonu
    gene_type=int,                      # Celobrojni geni
    gene_space=[0, 1],                  # 0 = preskoči kolonu, 1 = koristi kolonu
    fitness_func=fitness_func,          # Funkcija koja ocenjuje kombinaciju kolona
    parent_selection_type="tournament", # Izbor roditelja takmičenjem nasumično izabranih kandidata
    keep_elitism=1,                     # Najbolja kombinacija se čuva za sledeću generaciju
    crossover_type="single_point",      # Spajanje delova dva roditelja na jednoj tački preseka
    mutation_type="random",             # Nasumičan izbor dozvoljene vrednosti pri mutaciji
    mutation_num_genes=1,               # Jedan gen se bira za mutaciju kod svakog potomka
    random_seed=42                      # Ponovljivi rezultati pri istim podacima i podešavanjima
)

ga.run()

solution, fitness, solution_idx = ga.best_solution()
best_inputs = [column for column, gene in zip(inputs, solution) if gene == 1]

print("Najbolja pronađena kombinacija gena:", solution)
print("Izabrane kolone:", best_inputs)
print("Validacioni RMSE:", 1.0 / fitness - 1.0)

# Završno treniranje na 80% podataka
model = LinearRegression()
model.fit(X_dev[best_inputs], y_dev)

# Konačno testiranje na preostalih 20%
y_pred = model.predict(X_test[best_inputs])

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", rmse(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))