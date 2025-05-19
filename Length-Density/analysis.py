from pandas import read_csv, DataFrame
from numpy import mean, sqrt, pi, square
from io import TextIOWrapper
from datetime import datetime


def compute_uncertainty_of_length(
    source: DataFrame, label: str, ret: TextIOWrapper
) -> float:

    length_data_list: list[float] = [float(num) for num in source[label]]
    mean_of_length: float = mean(length_data_list)
    denominator: float = 1 / (len(length_data_list) - 1)

    S_of_length: float = sqrt(
        sum(
            [
                (x - mean_of_length) * (x - mean_of_length) * denominator
                for x in length_data_list
            ]
        )
    )
    pS_of_length: float = S_of_length / sqrt(len(length_data_list))

    uAx: float = 1.20 * pS_of_length
    uBx: float = 0.1 / sqrt(3)
    ux: float = sqrt(uAx * uAx + uBx * uBx)

    ret.writelines(
        [
            f"{label}:\n",
            f"average-{label}: {mean_of_length}\n",
            f"S-{label}: {S_of_length}\n",
            f"pS-{label}: {pS_of_length}\n",
            f"uAx-{label}: {uAx}\n",
            f"uBx-{label}: {uBx}\n",
            f"ux-{label}: {ux}\n" f"result of {label}: {mean_of_length} ({ux})\n" "\n",
        ]
    )

    return mean_of_length


def compute_uncertainty_of_diameter(
    source: DataFrame, label: str, ret: TextIOWrapper
) -> tuple[float, float]:

    diameter_data_list: list[float] = [float(num) for num in source[label]]
    mean_of_diameter: float = mean(diameter_data_list)
    denominator: float = 1 / (len(diameter_data_list) - 1)

    S_of_diameter: float = sqrt(
        sum([square(x - mean_of_diameter) * denominator for x in diameter_data_list])
    )
    pS_of_diameter: float = S_of_diameter / sqrt(len(diameter_data_list))

    uAx: float = 1.20 * pS_of_diameter
    uBx: float = 0.02 / sqrt(3)
    ux: float = sqrt(uAx * uAx + uBx * uBx)

    ret.writelines(
        [
            f"{label}:\n",
            f"average-{label}: {mean_of_diameter}\n",
            f"S-{label}: {S_of_diameter}\n",
            f"pS-{label}: {pS_of_diameter}\n",
            f"uAx-{label}: {uAx}\n",
            f"uBx-{label}: {uBx}\n",
            f"ux-{label}: {ux}\n"
            f"result of {label}: {mean_of_diameter} ({ux})\n"
            "\n",
        ]
    )

    return mean_of_diameter, ux


def compute_micrometer_screw(
    diameter_data_list: list[float], label: str, ret: TextIOWrapper
) -> tuple[float, float]:

    mean_of_diameter: float = mean(diameter_data_list)
    denominator: float = 1 / (len(diameter_data_list) - 1)

    S_of_diameter: float = sqrt(
        sum([square(x - mean_of_diameter) * denominator for x in diameter_data_list])
    )
    pS_of_diameter: float = S_of_diameter / sqrt(len(diameter_data_list))

    uAx: float = 1.11 * pS_of_diameter
    uBx: float = 0.001 / sqrt(3)
    ux: float = sqrt(uAx * uAx + uBx * uBx)

    ret.writelines(
        [
            f"{label}:\n",
            f"average-{label}: {mean_of_diameter}\n",
            f"S-{label}: {S_of_diameter}\n",
            f"pS-{label}: {pS_of_diameter}\n",
            f"uAx-{label}: {uAx}\n",
            f"uBx-{label}: {uBx}\n",
            f"ux-{label}: {ux}\n"
            f"result of {label}: {mean_of_diameter} ({ux})\n"
            "\n",
        ]
    )

    return mean_of_diameter, ux


def compute_mass_data(
    mass_data_list: list[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    label: str,
    ret: TextIOWrapper,
) -> None:

    def compute_each(
        data: tuple[float, float, float], label: str, ret: TextIOWrapper
    ) -> None:
        mean_of_data: float = mean(data)
        denominator: float = 1 / (len(data) - 1)

        S_of_data: float = sqrt(
            sum([square(x - mean_of_data) * denominator for x in data])
        )
        pS_of_data: float = S_of_data / sqrt(len(data))

        uAx: float = 1.32 * pS_of_data
        uBx: float = 0.01 / sqrt(3)
        ux: float = sqrt(uAx * uAx + uBx * uBx)

        ret.writelines(
            [
                f"{label}:\n",
                f"average-{label}: {mean_of_data}\n",
                f"S-{label}: {S_of_data}\n",
                f"pS-{label}: {pS_of_data}\n",
                f"uAx-{label}: {uAx}\n",
                f"uBx-{label}: {uBx}\n",
                f"ux-{label}: {ux}\n"
                f"result of {label}: {mean_of_data} ({ux})\n"
                "\n",
            ]
        )

        return mean_of_data, ux

    (m0, um0), (m1, um1), (m2, um2) = (
        compute_each(mass_data_list[0], "m0", ret),
        compute_each(mass_data_list[1], "m1", ret),
        compute_each(mass_data_list[2], "m2", ret),
    )

    ret.writelines(
        [
            f"p - {m0 / (m2 - m1)}\n",
            f"up - {1.21 * sqrt(square(um0 / m0) + square(um1 / m1) + square(um2 / m2))}\n",
            "\n",
        ]
    )

    return


def automatic(func):
    if __name__ == "__main__":
        return func()
    raise ResourceWarning


@automatic
def main() -> None:
    source: DataFrame = read_csv("data.csv")
    ret: TextIOWrapper = open("data.ret", "w")
    micrometer_screw_data: list[float] = [
        22.188,
        22.182,
        22.192,
        22.198,
        22.180,
        22.187,
    ]
    delta_micrometer_screw_data: float = 0.011
    mass_data: list[float] = [
        (4.92, 4.91, 4.90),
        (297.18, 297.17, 297.16),
        (301.23, 301.14, 301.12),
    ]

    ret.write(f"[{datetime.now()}]\n\n")

    for label in ["l1", "l2", "l3", "l4"]:
        compute_uncertainty_of_length(source, label, ret)

    (d1, ud1), (h1, uh1), (d2, ud2), (h2, uh2) = (
        compute_uncertainty_of_diameter(source, "D1", ret),
        compute_uncertainty_of_diameter(source, "H1", ret),
        compute_uncertainty_of_diameter(source, "D2", ret),
        compute_uncertainty_of_diameter(source, "H2", ret),
    )

    v: float = pi / 4 * (square(d1) * h1 - square(d2) * h2)
    dv: float = (
        pi
        / 4
        * sqrt(
            square(square(d1) * h1) * (square(uh1 / h1) + 4 * square(ud1 / d1))
            + square(square(d2) * h2) * (square(uh2 / h2) + 4 * square(ud2 / d2))
        )
    )
    ret.writelines(
        [
            f"V - {v}\n",
            f"dV - {dv}\n",
            "\n",
        ]
    )

    compute_micrometer_screw(micrometer_screw_data, "uncorrected", ret)
    d, ud = compute_micrometer_screw(
        [x - delta_micrometer_screw_data for x in micrometer_screw_data],
        "corrected",
        ret,
    )
    v: float = pi / 6 * square(d) * d
    uv: float = 3 * v * ud / d
    ret.writelines(
        [
            f"V - {v}\n",
            f"uv - {uv}\n",
            "\n",
        ]
    )
    
    compute_mass_data(mass_data, "mass", ret)

    ret.flush()
    ret.close()
