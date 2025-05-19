from datetime import datetime
from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def automatic(func) -> any:
    if __name__ == "__main__":
        return func()
    raise ResourceWarning


def draw_linear(path: str) -> None:
    # Initialize data.
    csv: DataFrame = read_csv(path)
    e_list: list[float] = [float(number) for number in csv["E"]]
    e_array: list[list[float]] = [[number] for number in e_list]
    r_list: list[float] = [float(number) for number in csv["R0"]]
    dr_list: list[float] = [float(number) for number in csv["DR0"]]
    di_list: list[float] = [float(number) for number in csv["DI"]]

    # Compute
    def compute(
        r_list: list[float], dr_list: list[float], di_list: list[float]
    ) -> list[float]:
        s_array: list[float] = [di for di in di_list]
        for i in range(len(s_array)):
            s_array[i] *= r_list[i] / dr_list[i]
        return s_array

    s_array: list[float] = compute(r_list, dr_list, di_list)

    # Set linear regression
    model: LinearRegression = LinearRegression()
    model.fit(e_array, s_array)
    for_prediction: list[list[float]] = []
    num: float = -0.05
    while num < max(e_list) + 0.05:
        for_prediction.append([num])
        num += 0.02
    prediction: list[float] = model.predict(for_prediction)

    # Init.
    plt.figure(figsize=(8, 6))
    plt.title(f"{path.removesuffix('.csv')}")
    plt.xlabel("E/V", loc="right")
    plt.ylabel("S/nA", loc="top")
    plt.grid(True)

    # Draw predicted.
    plt.plot(for_prediction, prediction, label="regression", color="blue")

    # Draw experimental.
    plt.scatter(e_list, s_array, label="experiment", color="red")
    for i in range(len(e_list)):
        plt.annotate(
            f"{s_array[i]:.2f}",
            xy=(e_list[i], s_array[i]),
            xytext=(e_list[i] + 0.05, s_array[i]),
            fontsize=6,
            ha="left",
            va="bottom",
        )

    # Output.
    plt.legend(loc="lower right")
    plt.savefig(
        f"linear_{path.removesuffix('.csv')}.png",
        format="png",
        bbox_inches="tight",
        dpi=480,
    )
    with open(f"{path.removesuffix('.csv')}.ret", "a+") as result:
        if result.tell() != 0:
            result.write(f"\n[{datetime.now()} figure output]\n")
        else:
            result.write(f"[{datetime.now()} figure output]\n")
        result.write(".LINEAR\n")
        result.write(f"> r: {model.score(e_array, s_array):.2f}\n")
        result.write(f"> coef: {model.coef_}\n")
        result.write(f"> intercept: {model.intercept_}\n")


@automatic
def main() -> None:
    draw_linear("E-S.csv")
