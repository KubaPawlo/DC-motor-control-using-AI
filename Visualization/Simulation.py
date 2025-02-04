import json
import numpy as np
import matplotlib.pyplot as plt

class DCMotor:
    def __init__(self, Ua=24, Ra=4.3, La=0.0015, k=0.05, J=1.03e-5, m_op=0.06, dt=0.001):
        self.Ua = Ua  # Napięcie zasilania (będzie nadpisywane przy każdym kroku)
        self.Ra = Ra  # Rezystancja w obwodzie twornika
        self.La = La  # Indukcyjność w obwodzie twornika
        self.k = k    # Stała momentu (Nm/A) i stała SEM (V/(rad/s))
        self.J = J    # Moment bezwładności [kg·m^2]
        self.m_op = m_op  # Moment oporu [Nm]
        self.dt = dt  # Krok czasowy
        self.I_max = 3.9  # Maksymalny prąd
        self.reset()

    def reset(self):
        self.predkosc = 0    # RPM
        self.w = 0           # rad/s
        self.ia = 0          # prąd twornika [A]
        self.w_dot_prev = 0
        self.ia_dot_prev = 0

    def set_load(self, load):
        self.m_op = load

    def step(self):
        # Równanie prądu twornika
        self.ia_dot = (self.Ua - self.k*self.w - self.ia*self.Ra) / self.La
        # Całkowanie prądu metodą trapezów
        self.ia += 0.5 * (self.ia_dot + self.ia_dot_prev) * self.dt
        # Ograniczenie prądu
        self.ia = np.clip(self.ia, -self.I_max, self.I_max)
        self.ia_dot_prev = self.ia_dot

        # Równanie momentu na wale -> przyspieszenie kątowe
        self.w_dot = (self.k*self.ia - self.m_op) / self.J
        # Ograniczenie przyśpieszenia
        self.w_dot = np.clip(self.w_dot, -1e4, 1e4)

        # Aktualizacja prędkości kątowej
        self.w += 0.5 * (self.w_dot + self.w_dot_prev) * self.dt
        self.w_dot_prev = self.w_dot

        # Ograniczenie max prędkości
        max_rpm = 5000
        max_w = max_rpm * 2*np.pi / 60
        self.w = np.clip(self.w, -max_w, max_w)

        # Przeliczenie rad/s na RPM
        self.predkosc = self.w * 60 / (2*np.pi)

def simulate_motor(voltage_sequence, J=1.03e-5, load=0.06, dt=0.001):

    motor = DCMotor(J=J, m_op=load, dt=dt)
    speeds = []

    for volt in voltage_sequence:
        motor.Ua = volt
        motor.step()
        speeds.append(motor.predkosc)

    return speeds

def main():
    # Pliki do wczytania
    json_files = [
        "voltage_data_RPM500_Moment0.00e+00.json",
        "voltage_data_RPM850_Moment0.00e+00.json",
        "voltage_data_RPM1150_Moment0.00e+00.json"
    ]

    time = np.arange(3000) * 0.001

    for idx, filename in enumerate(json_files):
        with open(filename, "r") as f:
            data = json.load(f)

        rpm_info = data.get("rpm", None)
        inertia = data.get("moment_bezwladnosci", 1.03e-5)
        volt_seq = data.get("voltage_sequence", [])

        speeds = simulate_motor(volt_seq, J=inertia, load=0.06, dt=0.001)

        # Rysowanie wykresu
        plt.figure(figsize=(8,4))
        plt.plot(time, speeds, label=f"Symulacja z pliku: {filename}")
        plt.xlabel("Czas [s]")
        plt.ylabel("Prędkość [RPM]")
        title_txt = f"Wykres prędkości - {filename}"
        if rpm_info is not None:
            title_txt += f" (json RPM={rpm_info})"
        plt.title(title_txt)
        plt.legend()
        plt.grid(True)

        out_png = f"motor_speed_{rpm_info}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Zapisano wykres: {out_png}")

if __name__ == "__main__":
    main()