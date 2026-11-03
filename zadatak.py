import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def rmse(actual, predictions):
    actual = np.asarray(actual).ravel()
    predictions = np.asarray(predictions).ravel()
    return np.sqrt(np.mean((actual - predictions) ** 2))


def plot_predictions(actual, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(np.asarray(actual).ravel(), label="Stvarna potrošnja")
    plt.plot(np.asarray(predictions).ravel(), color="red", alpha=0.8, label="Predviđena potrošnja")
    plt.xlabel("Merenje u test skupu")
    plt.ylabel("Zone 1 Power Consumption")
    plt.legend()
    plt.tight_layout()
    plt.show()


df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

inputs = ["Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows"]
target = "Zone 1 Power Consumption"

df[inputs + [target]] = df[inputs + [target]].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=inputs + [target])

X = df[inputs]
y = df[target]

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, shuffle=False
)


def fitness_func(ga_instance, solution, solution_idx):
    selected_columns = [column for column, gene in zip(inputs, solution) if gene == 1]

    if not selected_columns:
        return 0.0

    candidate_model = LinearRegression()
    candidate_model.fit(X_train[selected_columns], y_train)
    predictions = candidate_model.predict(X_val[selected_columns])
    mae = mean_absolute_error(y_val, predictions)

    return 1.0 / (1.0 + rmse(y_val, predictions))


ga = pygad.GA(
    num_generations=30,                 # Broj generacija
    sol_per_pop=20,                     # Broj kombinacija kolona u svakoj generaciji
    num_parents_mating=4,               # Broj roditelja koji stvaraju potomke
    num_genes=len(inputs),              # Jedan gen za svaku ulaznu kolonu
    gene_type=int,                     # Celobrojni geni
    gene_space=[0, 1],                  # 0 = preskoči kolonu, 1 = koristi kolonu
    fitness_func=fitness_func,          # Funkcija koja ocenjuje kombinaciju kolona
    parent_selection_type="tournament", # Izbor roditelja takmičenjem nasumično izabranih kandidata
    keep_elitism=1,                     # Najbolja kombinacija se čuva za sledeću generaciju
    crossover_type="single_point",      # Spajanje delova dva roditelja na jednoj tački preseka
    mutation_type="random",             # Nasumičan izbor dozvoljene vrednosti pri mutaciji
    mutation_num_genes=1,               # Jedan gen se bira za mutaciju kod svakog potomka
    random_seed=42                     # Ponovljivi rezultati pri istim podacima i podešavanjima
)

ga.run()

solution, fitness, solution_idx = ga.best_solution()
best_inputs = [column for column, gene in zip(inputs, solution) if gene == 1]

print("Najbolja pronađena kombinacija gena:", solution)
print("Izabrane kolone:", best_inputs)
print("Validacioni MAE:", 1.0 / fitness - 1.0)

model = LinearRegression()
model.fit(X_dev[best_inputs], y_dev)

y_pred = model.predict(X_test[best_inputs])

print("\nRezultati na test skupu:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", rmse(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

plot_predictions(y_test, y_pred)