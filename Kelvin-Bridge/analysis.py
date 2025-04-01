# author: kleedaisuki
# encoding: UTF-8
# using Python 3.12.0

from datetime import datetime
from numpy import mean, std, sqrt, pi
from scipy.stats import pearsonr
from json import load


def automatic(func) -> None:
    if __name__ == "__main__":
        return func()
    raise ResourceWarning


def compute_length(ret_file: str, length: float) -> tuple[float, float]:
    with open(ret_file, "a") as ret:
        ret.write(f"> length: {round(length, 2):.2f}\n")
    return (length * 0.01, (0.017 / length) ** 2)


def compute_diameter(ret_file: str, diameter_data: list[float]) -> tuple[float, float]:
    diameter: float = mean(diameter_data)
    n: int = len(diameter_data)

    u_ad: float = 1.14 * std(diameter_data) / sqrt(n * (n - 1))
    u_bd: float = 0.0005773502691896258  # Pre-computed.
    u_d: float = sqrt(u_ad * u_ad + u_bd * u_bd)

    with open(ret_file, "a") as ret:
        ret.write(f"> u_ad: {round(u_ad, 4):.4f}\n")
        ret.write(f"> u_bd: {round(u_bd, 4):.4f}\n")
        ret.write(f"> diameter: {round(diameter, 4):.4f} +- {round(u_d, 4)}\n")

    return (diameter * 0.001, (u_d / diameter) ** 2 * 4)


def compute_resistance_and_sensitivity(
    ret_file: str, r2: float, dr_list: list[float], di_list: list[float]
) -> tuple[float, float]:
    dr: float = mean(dr_list)
    di: float = mean(di_list)
    sensitivity: float = r2 * di / dr

    px: float = sqrt(3.0004e-06 + (0.1 / sensitivity))  # Pre-computed.
    Rx: float = 1e-06 * r2  # Pre-computed.
    u_r: float = px * Rx

    # Check.
    correlation, _ = pearsonr(dr_list, di_list)

    with open(ret_file, "a") as ret:
        ret.write(f"> dr: {round(dr, 1):.1f}\n")
        ret.write(f"> di: {round(di, 1):.1f}\n")
        ret.write(f"> R^2: {correlation}\n")
        ret.write(f"> px: {round(px * 1000, 1):.1f}e-3\n")
        ret.write(f"> u_r: {round(u_r * 10000, 4):.4f}e-4\n")
        ret.write(f"> S: {round(sensitivity, 2):.2f}\n")
        ret.write(f"> Rx: {round(Rx * 10000, 4):.4f}e-4")
        ret.write(f" +- {round(u_r * 10000, 4):.4f}e-4\n")

    return (Rx, (u_r / Rx) ** 2)


def compute_resistivity(
    ret_file: str, length: float, diameter: float, Rx: float, u_a: float
) -> None:
    resistivity: float = pi / 4 * Rx * diameter**2 / length
    u_resi: float = u_a * resistivity
    with open(ret_file, "a") as ret:
        ret.write(f"> u_resi: {round(u_resi * 10000000, 3):.3f}e-8\n")
        ret.write(f"> Resistivity: {round(resistivity * 10000000, 4):.4f}e-8\n")


@automatic
def main() -> None:
    data_file: any
    with open("data.json", "r") as j:
        data_file = load(j)
    for obj in ("Cu", "Al", "Fe"):
        data: dict[any] = data_file[obj]
        with open(f"{obj}.ret", "w") as ret:
            ret.write(f"[{datetime.now()}] Analysis:\n")

        length, part_l = compute_length(f"{obj}.ret", float(data["length"]))
        diameter, part_d = compute_diameter(
            f"{obj}.ret", [float(x) for x in data["diameter_data"]]
        )

        r2: float = float(data["resistance"])
        dr_list: list[float] = []
        di_list: list[float] = []
        sensitivity_data = data["sensitivity_data"]
        for i in range(len(sensitivity_data)):
            dr_list.append(float(sensitivity_data[i][0]))
            di_list.append(float(sensitivity_data[i][1]))
        Rx, part_rx = compute_resistance_and_sensitivity(
            f"{obj}.ret", r2, dr_list, di_list
        )

        compute_resistivity(
            f"{obj}.ret", length, diameter, Rx, sqrt(part_l + part_d + part_rx)
        )
