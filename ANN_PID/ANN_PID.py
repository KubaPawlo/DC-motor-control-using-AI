import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# KLASY SILNIKA I REGULATORA PID (do symulacji)


class DCMotor:

    def __init__(self, Ua=0.0, Ra=4.3, La=0.0015, k=0.05,
                 J=1.03e-5, m_op=0.06, dt=0.001):

        self.Ua = Ua
        self.Ra = Ra
        self.La = La
        self.k = k
        self.J = J
        self.m_op = m_op  # Stały moment oporu
        self.dt = dt
        self.I_max = 3.9  # ograniczenie prądu (A)

        self.reset()

    def reset(self):

        self.predkosc = 0.0    # prędkość w RPM
        self.w = 0.0           # prędkość kątowa w rad/s
        self.ia = 0.0          # prąd w tworniku
        self.w_dot_prev = 0.0
        self.ia_dot_prev = 0.0

    def step(self):

        # Równanie prądu
        ia_dot = (self.Ua - self.k * self.w - self.ia * self.Ra) / self.La
        self.ia += (ia_dot + self.ia_dot_prev) / 2.0 * self.dt
        self.ia = np.clip(self.ia, -self.I_max, self.I_max)
        self.ia_dot_prev = ia_dot

        # Równanie momentu - przyspieszenie kątowe
        w_dot = (self.k * self.ia - self.m_op) / self.J
        self.w += (w_dot + self.w_dot_prev) / 2.0 * self.dt
        self.w_dot_prev = w_dot

        # Ograniczenie RPM
        max_rpm = 5000
        w_limit = max_rpm * 2.0 * np.pi / 60.0
        self.w = np.clip(self.w, -w_limit, w_limit)

        # Przeliczenie w na RPM
        self.predkosc = self.w * 60.0 / (2.0 * np.pi)


class PIDController:

    def __init__(self, kp=0.1, ki=0.01, kd=0.01, predkosc_zadana=1500):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.predkosc_zadana = predkosc_zadana

        self.e_sum = 0.0
        self.e_last = 0.0
        self.u_max = 24.0
        self.u_min = -24.0

    def set_voltage_limits(self, u_min, u_max):
        self.u_min = u_min
        self.u_max = u_max

    def calculate(self, predkosc_aktualna, dt=0.001):

        e_current = self.predkosc_zadana - predkosc_aktualna
        self.e_sum += e_current * dt

        # anti-windup
        if self.ki != 0:
            integral_term = self.ki * self.e_sum
            if integral_term > self.u_max:
                self.e_sum = self.u_max / self.ki
            elif integral_term < self.u_min:
                self.e_sum = self.u_min / self.ki

        de = (e_current - self.e_last) / dt
        self.e_last = e_current

        u = self.kp * e_current + self.ki * self.e_sum - self.kd * de
        return float(np.clip(u, self.u_min, self.u_max))



# CALLBACK DO WYKRESÓW GRADIENTÓW

class GradientPlotCallback(tf.keras.callbacks.Callback):

    def __init__(self, my_model, X_sample, Y_sample, outdir="plots/gradients"):
        super().__init__()
        self.my_model = my_model
        self.X_sample = tf.constant(X_sample, dtype=tf.float32)
        self.Y_sample = tf.constant(Y_sample, dtype=tf.float32)
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch % 5 == 0):
            self.plot_gradient_for_epoch(epoch)

    def plot_gradient_for_epoch(self, epoch):
        with tf.GradientTape() as tape:
            preds = self.my_model(self.X_sample, training=True)
            loss_value = self.my_model.compiled_loss(
                self.Y_sample,
                preds,
                regularization_losses=self.my_model.losses
            )
        gradients = tape.gradient(loss_value, self.my_model.trainable_variables)

        kernel_norms = []
        kernel_names = []
        bias_norms = []
        bias_names = []

        for var, grad in zip(self.my_model.trainable_variables, gradients):
            if grad is None:
                continue
            gnorm = tf.sqrt(tf.reduce_sum(tf.square(grad))).numpy()
            if 'kernel' in var.name:
                kernel_norms.append(gnorm)
                kernel_names.append(var.name)
            elif 'bias' in var.name:
                bias_norms.append(gnorm)
                bias_names.append(var.name)

        plt.figure(figsize=(10, 4))
        # Kernels
        plt.subplot(1, 2, 1)
        plt.bar(range(len(kernel_norms)), kernel_norms, alpha=0.7, color="blue")
        plt.xticks(range(len(kernel_norms)), kernel_names, rotation=90, fontsize=8)
        plt.ylabel("Gradient L2 norm (kernel)")
        plt.title(f"Kernel grads (epoch={epoch})")
        plt.tight_layout()

        # Bias
        plt.subplot(1, 2, 2)
        plt.bar(range(len(bias_norms)), bias_norms, alpha=0.7, color="green")
        plt.xticks(range(len(bias_norms)), bias_names, rotation=90, fontsize=8)
        plt.ylabel("Gradient L2 norm (bias)")
        plt.title(f"Bias grads (epoch={epoch})")
        plt.tight_layout()

        filename = os.path.join(self.outdir, f"gradient_epoch_{epoch}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[GradientPlotCallback] Zapisano wykres gradientów epoka={epoch}: {filename}")



# CALLBACK DO ZAPISYWANIA MSE W JSON

class MSEHistoryCallback(tf.keras.callbacks.Callback):

    def __init__(self, json_filename="mse_history.json"):
        super().__init__()
        self.json_filename = json_filename
        self.history_dict = {
            "epoch": [],
            "loss": [],
            "val_loss": []
        }

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        current_val_loss = logs.get("val_loss")
        self.history_dict["epoch"].append(epoch)
        self.history_dict["loss"].append(float(current_loss) if current_loss else None)
        self.history_dict["val_loss"].append(float(current_val_loss) if current_val_loss else None)

    def on_train_end(self, logs=None):
        with open(self.json_filename, "w") as f:
            json.dump(self.history_dict, f, indent=2)
        print(f"[MSEHistoryCallback] Zapisano historię MSE do: {self.json_filename}")



# WCZYTANIE DANYCH Z JSON

def load_data_from_json(json_filename):

    with open(json_filename, "r") as f:
        data = json.load(f)

    X_list = []
    Y_list = []
    W_list = []

    for key, val in data.items():
        rpm = val.get("rpm", None)
        mom = val.get("moment_bezwladnosci", None)
        kp = val.get("kp", None)
        ki = val.get("ki", None)
        kd = val.get("kd", None)
        fit = val.get("fitness", None)

        if None in [rpm, mom, kp, ki, kd, fit]:
            continue

        X_list.append([rpm, mom])       # X -> [RPM, J]
        Y_list.append([kp, ki, kd])     # Y -> [kp, ki, kd]

        # Waga: 1/(fitness + 1) ograniczona do 50
        w_raw = 1.0 / (fit + 1.0)
        w = min(w_raw, 50.0)
        W_list.append(w)

    X_array = np.array(X_list, dtype=np.float32)
    Y_array = np.array(Y_list, dtype=np.float32)
    W_array = np.array(W_list, dtype=np.float32)

    return X_array, Y_array, W_array



# BUDOWA MODELU SIECI

def build_mlp_model(input_dim=2, hidden_units=128):
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(hidden_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(hidden_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Wyjście: 3 parametry (kp, ki, kd)
    model.add(Dense(3, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model



# FUNKCJE RYSUJĄCE HISTORIĘ UCZENIA ORAZ SCATTERY PRED VS. ACT

def plot_training_history(history, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Przebieg uczenia sieci')
    plt.legend()
    plt.grid(True)
    fname = os.path.join(outdir, "training_loss.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Zapisano wykres historii uczenia: {fname}")


def plot_predictions(model, X, Y, dataset_name="train", outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    Y_pred = model.predict(X)
    param_names = ["kp", "ki", "kd"]

    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(Y[:, i], Y_pred[:, i], alpha=0.5, c='b', label=dataset_name)
        mi = min(Y[:, i].min(), Y_pred[:, i].min())
        ma = max(Y[:, i].max(), Y_pred[:, i].max())
        plt.plot([mi, ma], [mi, ma], 'r--', label="Ideal diag")
        plt.xlabel("Actual " + param_names[i])
        plt.ylabel("Predicted " + param_names[i])
        plt.title(f"{param_names[i]} ({dataset_name})")
        plt.grid()
        plt.legend()

    fname = os.path.join(outdir, f"pred_scatter_{dataset_name}.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Zapisano wykres predykcji {dataset_name}: {fname}")



# FUNKCJE DO SYMULACJI

def simulate_motor(kp, ki, kd, rpm_target, inertia_j, sim_time=3.0, dt=0.001):

    motor = DCMotor(J=inertia_j, dt=dt, m_op=0.06)  # ewentualnie zmień m_op
    pid = PIDController(kp=kp, ki=ki, kd=kd, predkosc_zadana=rpm_target)
    pid.set_voltage_limits(-24, 24)

    steps = int(sim_time / dt)
    time_list = []
    rpm_list = []

    for i in range(steps):
        t = i * dt
        u = pid.calculate(motor.predkosc, dt=dt)
        motor.Ua = u
        motor.step()

        time_list.append(t)
        rpm_list.append(motor.predkosc)

    return np.array(time_list), np.array(rpm_list)


def simulate_and_plot(kp_real, ki_real, kd_real,
                      kp_pred, ki_pred, kd_pred,
                      rpm_target, inertia_j, idx,
                      sim_time=3.0, dt=0.001, outdir="plots/simulations"):

    os.makedirs(outdir, exist_ok=True)

    # Symulacja z REAL PID
    t1, rpm1 = simulate_motor(kp_real, ki_real, kd_real, rpm_target, inertia_j, sim_time, dt)
    # Symulacja z PRED PID
    t2, rpm2 = simulate_motor(kp_pred, ki_pred, kd_pred, rpm_target, inertia_j, sim_time, dt)

    plt.figure(figsize=(8, 5))
    plt.plot(t1, rpm1, label="Real PID (z JSON)", color='blue')
    plt.plot(t2, rpm2, label="Pred PID (z sieci)", color='red', linestyle='--')

    plt.title(f"Symulacja: próbka testowa {idx}")
    plt.xlabel("Czas [s]")
    plt.ylabel("Prędkość obrotowa [RPM]")
    plt.grid(True)
    plt.legend()


    text_str = (f"Real PID: kp={kp_real:.3f}, ki={ki_real:.3f}, kd={kd_real:.3f}\n"
                f"NN   PID: kp={kp_pred:.3f}, ki={ki_pred:.3f}, kd={kd_pred:.3f}\n"
                f"rpm_target={rpm_target:.1f}, moment_bezwladnosci={inertia_j:.3e}")

    plt.subplots_adjust(bottom=0.3)
    plt.text(0.5, 0.02, text_str, ha='center', va='bottom',
             transform=plt.gca().transAxes,
             fontsize=9)

    filename = os.path.join(outdir, f"simulation_test_{idx}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[simulate_and_plot] Zapisano wykres symulacji: {filename}")


def simulate_test_examples(model, X_test, Y_test, scaler,
                           sim_time=3.0, dt=0.001, outdir="plots/simulations"):

    os.makedirs(outdir, exist_ok=True)
    Y_pred = model.predict(X_test)
    X_unscaled = scaler.inverse_transform(X_test)  # wracamy do [rpm, J] w oryg. skali

    for i in range(len(X_test)):
        rpm_target = X_unscaled[i, 0]
        inertia_j  = X_unscaled[i, 1]

        kp_real, ki_real, kd_real = Y_test[i]
        kp_pred, ki_pred, kd_pred = Y_pred[i]

        simulate_and_plot(kp_real, ki_real, kd_real,
                          kp_pred, ki_pred, kd_pred,
                          rpm_target, inertia_j,
                          idx=i,
                          sim_time=sim_time,
                          dt=dt,
                          outdir=outdir)



# GŁÓWNA FUNKCJA main

def main():
    # Wczytanie danych
    json_file = "results.json"
    X, Y, W = load_data_from_json(json_file)
    print(f"Wczytano {len(X)} przykładów z pliku {json_file}.")

    if len(X) == 0:
        print("Brak danych do trenowania! Sprawdź plik JSON.")
        return

    # Normalizacja X (rpm, moment_bezwladnosci) -> [0,1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Podział na train i test
    X_train, X_test, Y_train, Y_test, W_train, W_test = \
        train_test_split(X_scaled, Y, W, test_size=0.2, random_state=42)
    print(f"Trening na {len(X_train)} przykładach, test na {len(X_test)}.")

    # Budowa modelu MLP
    model = build_mlp_model(input_dim=X_train.shape[1], hidden_units=128)

    # Callbacki
    sample_size = min(16, len(X_train))  # do liczenia gradientów
    X_sample = X_train[:sample_size]
    Y_sample = Y_train[:sample_size]
    grad_plot_cb = GradientPlotCallback(my_model=model, X_sample=X_sample, Y_sample=Y_sample)
    mse_history_cb = MSEHistoryCallback(json_filename="mse_history.json")

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Trening z wagami W
    history = model.fit(
        X_train, Y_train,
        sample_weight=W_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        verbose=1,
        callbacks=[grad_plot_cb, mse_history_cb, early_stop, reduce_lr]
    )

    # Wykres przebiegu uczenia
    plot_training_history(history, outdir="plots")

    # Ocena na zbiorze testowym
    test_loss = model.evaluate(X_test, Y_test, sample_weight=W_test)
    print(f"Test MSE = {test_loss:.4f}")

    # Wykres predykcji (scatter predicted vs actual)
    plot_predictions(model, X_train, Y_train, dataset_name="train", outdir="plots")
    plot_predictions(model, X_test, Y_test, dataset_name="test", outdir="plots")

    # Zapis modelu
    model.save("pid_model.h5")
    print("Zapisano model 'pid_model.h5'.")

    # Symulacje prędkości obrotowej na zbiorze testowym i zapis wykresów
    simulate_test_examples(model, X_test, Y_test, scaler,
                           sim_time=3.0, dt=0.001,
                           outdir="plots/simulations")

    print("=== Koniec uczenia. Pliki PNG zapisano w folderach 'plots' oraz 'plots/simulations'. ===")


if __name__ == "__main__":
    main()

