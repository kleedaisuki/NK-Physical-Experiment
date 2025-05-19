from datetime import datetime
from pandas import read_csv, DataFrame
from numpy import sqrt, array, ndarray
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


def analysis_U_I_data(path: str) -> None:
    # Initialize data.
    csv: DataFrame = read_csv(path)
    U_list: list[float] = [float(number) for number in csv["U"]]
    I_list: list[float] = [float(number) * 0.001 for number in csv["I"]]

    # Computation.
    u1: float = U_list[0]
    u2: float = U_list[len(U_list) - 1]
    i1: float = I_list[0]
    i2: float = I_list[len(U_list) - 1]

    r: float = (u2 - u1) / (i2 - i1 - (u2 - u1) / 1e7)
    delta_u: float = 0.02 * 0.01 * max(U_list) + 4 * 0.0001
    delta_i: float = 1.2 * 0.01 * max(I_list) * 1000 + 3 * 0.01
    px: float = sqrt((delta_u / (u2 - u1)) ** 2 + (delta_i * 0.001 / (i2 - i1)) ** 2)

    # Write results to file.
    with open(f"{path.removesuffix(".csv")}.ret", "a+") as result:
        if result.tell() != 0:
            result.write(f"\n[{datetime.now()} statistics output]\n")
        else:
            result.write(f"[{datetime.now()} statistics output]\n")
        result.write(f"> R: {round(r, 2):.2f}\n")
        result.write(f"> delta_U: +- {round(delta_u, 4):.4f}V\n")
        result.write(f"> delta_I: +- {round(delta_i, 2):.2f}mA\n")
        result.write(f"> px: {round(px * 100, 2):.2f}%\n")
        result.write(f"Result: {round(r, 2):.2f} +- {round(px * r, 2):.2f}\n")


def draw_linear(path: str) -> None:
    # Initialize data.
    csv: DataFrame = read_csv(path)
    U_list: list[float] = [float(number) for number in csv["U"]]
    U_array: ndarray[ndarray[float]] = array([[number] for number in U_list])
    I_array: ndarray[float] = array([float(number) for number in csv["I"]])

    # Set linear regression
    model: LinearRegression = LinearRegression()
    model.fit(U_array, I_array)
    for_prediction: list[list[float]] = []
    num: float = -0.05
    while num < max(U_array) + 0.05:
        for_prediction.append([num])
        num += 0.02
    prediction: ndarray[float] = model.predict(for_prediction)

    # Init.
    plt.figure(figsize=(8, 6))
    plt.title(f"{path.removesuffix(".csv")}")
    plt.xlabel("U/V", loc="right")
    plt.ylabel("I/mA", loc="top")
    plt.grid(True)

    # Draw predicted.
    plt.plot(for_prediction, prediction, label="regression", color="blue")

    # Draw experimental.
    plt.scatter(U_list, I_array, label="experiment", color="red")
    for i in range(len(U_list)):
        plt.annotate(
            f"{I_array[i]:.2f}",
            xy=(U_list[i], I_array[i]),
            xytext=(U_list[i] + 0.05, I_array[i]),
            fontsize=6,
            ha="left",
            va="bottom",
        )

    # Output.
    plt.legend(loc="lower right")
    plt.savefig(
        f"linear_{path.removesuffix(".csv")}.png",
        format="png",
        bbox_inches="tight",
        dpi=480,
    )
    with open(f"{path.removesuffix(".csv")}.ret", "a+") as result:
        if result.tell() != 0:
            result.write(f"\n[{datetime.now()} figure output]\n")
        else:
            result.write(f"[{datetime.now()} figure output]\n")
        result.write(".LINEAR\n")
        result.write(f"> r: {model.score(U_array, I_array):.2f}\n")
        result.write(f"> coef: {model.coef_}\n")
        result.write(f"> intercept: {model.intercept_}\n")


def draw(path: str) -> None:
    # Initialize data.
    csv: DataFrame = read_csv(path)
    U_list: list[float] = [float(number) for number in csv["U"]]
    U_array: ndarray[ndarray[float]] = array([[number] for number in U_list])
    I_array: ndarray[float] = array([float(number) for number in csv["I"]])

    # Find best parameters for Support Vector Regression.
    param_grid: dict[str : list[float]] = {
        "C": [175, 200, 225],
        "gamma": [0.7, 1, 1.2],
        "epsilon": [0.55, 0.6, 0.63],
    }
    grid_search: GridSearchCV = GridSearchCV(
        SVR(kernel="rbf"), param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(U_array, I_array)

    # Get SVR.
    model: SVR = SVR(kernel="rbf", **grid_search.best_params_)
    model.fit(U_array, I_array)
    for_prediction: list[list[float]] = []
    num: float = -0.05
    while num < max(U_array) + 0.05:
        for_prediction.append([num])
        num += 0.02
    prediction: ndarray[float] = model.predict(for_prediction)

    # Init.
    plt.figure(figsize=(8, 6))
    plt.title(f"{path.removesuffix(".csv")}")
    plt.xlabel("U/V", loc="right")
    plt.ylabel("I/mA", loc="top")
    plt.grid(True)

    # Draw predicted.
    plt.plot(for_prediction, prediction, label="regression", color="blue")

    # Draw experimental.
    plt.scatter(U_list, I_array, label="experiment", color="red")
    for i in range(len(U_list)):
        plt.annotate(
            f"{I_array[i]:.2f}",
            xy=(U_list[i], I_array[i]),
            xytext=(U_list[i] + 0.05, I_array[i]),
            fontsize=6,
            ha="left",
            va="bottom",
        )

    # Output.
    plt.legend(loc="lower right")
    plt.savefig(
        f"svr_{path.removesuffix(".csv")}.png",
        format="png",
        bbox_inches="tight",
        dpi=480,
    )
    with open(f"{path.removesuffix(".csv")}.ret", "a+") as result:
        if result.tell() != 0:
            result.write(f"\n[{datetime.now()} figure output]\n")
        else:
            result.write(f"[{datetime.now()} figure output]\n")
        result.write(".SVR(RBF)\n")
        result.write(f"> r: {model.score(U_array, I_array):.2f}\n")
        result.write(f"Parameters:\n    {grid_search.best_params_}\n")


def automatic(function) -> any:
    if __name__ == "__main__":
        return function()
    raise ResourceWarning


@automatic
def main() -> None:
    analysis_U_I_data("Rx.csv")
    draw_linear("Rx.csv")
    draw("Dx.csv")
