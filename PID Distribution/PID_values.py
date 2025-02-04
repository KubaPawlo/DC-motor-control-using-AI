import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_pid_values(json_filename):

    #Zwraca trzy listy: kp_list, ki_list, kd_list
    with open(json_filename, "r") as f:
        data = json.load(f)

    kp_list = []
    ki_list = []
    kd_list = []

    for key, val in data.items():
        kp = val.get("kp", None)
        ki = val.get("ki", None)
        kd = val.get("kd", None)

        if kp is not None and ki is not None and kd is not None:
            kp_list.append(kp)
            ki_list.append(ki)
            kd_list.append(kd)

    return kp_list, ki_list, kd_list


def compute_percentage_distribution(values, bin_edges):

    # Jaki procent wartości `values` należy do kolejnych przedziałów
    counts, _ = np.histogram(values, bins=bin_edges)
    total = len(values)
    percentages = (counts / total) * 100.0
    return counts, percentages


def plot_distribution(percentages, bin_edges, param_name="kp", outdir="plots"):

    os.makedirs(outdir, exist_ok=True)

    # Przygotowanie etykiet na osi X, np. "0.00-0.05", "0.05-0.10"
    labels = []
    for i in range(len(bin_edges)-1):
        labels.append(f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}")

    plt.figure(figsize=(10,5))
    x_positions = np.arange(len(percentages))
    plt.bar(x_positions, percentages, width=0.8, alpha=0.7, color="blue")

    plt.xticks(x_positions, labels, rotation=45)
    plt.xlabel("Przedziały wartości")
    plt.ylabel("Procent wystąpień [%]")
    plt.title(f"Procentowy rozkład wartości {param_name}")

    plt.tight_layout()
    filename = os.path.join(outdir, f"distribution_{param_name}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Zapisano wykres: {filename}")


def main():
    json_file = "results.json"  # <-- tu wstaw swoją nazwę pliku JSON
    kp_list, ki_list, kd_list = load_pid_values(json_file)
    print(f"Wczytano {len(kp_list)} rekordów (kp, ki, kd).")


    bin_edges = np.arange(0.0, 1.0001, 0.05)

    # Obliczanie rozkładu (procenty) dla kp
    _, kp_percent = compute_percentage_distribution(kp_list, bin_edges)

    plot_distribution(kp_percent, bin_edges, param_name="kp", outdir="plots")

    # Dla ki
    _, ki_percent = compute_percentage_distribution(ki_list, bin_edges)
    plot_distribution(ki_percent, bin_edges, param_name="ki", outdir="plots")

    # Dla kd
    _, kd_percent = compute_percentage_distribution(kd_list, bin_edges)
    plot_distribution(kd_percent, bin_edges, param_name="kd", outdir="plots")


if __name__ == "__main__":
    main()