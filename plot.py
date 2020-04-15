import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv("data.csv", names=["alg", "vx", "vy", "error", "time"])

    brut = data.loc[(data["alg"] == "brut")]
    startbrut = data.loc[(data["alg"] == "startbrut")]
    indipendent = data.loc[(data["alg"] == "indipendent")]
    gradient = data.loc[(data["alg"] == "gradient")]


    batch = pd.DataFrame({"batch": range(4, 20, 2)})
    print(batch["batch"])

    # Data for plotting
    fig, ax = plt.subplots()
    ax.plot(batch["batch"], startbrut["time"], marker='o', linewidth=2, label="Start + Bruteforce")
    # ax.plot(batch["batch"], brut["time"], marker='o', label="Bruteforce")
    ax.plot(batch["batch"], indipendent["time"], marker='o', linewidth=2, label="Indipendent opt")
    ax.plot(batch["batch"], gradient["time"], marker='o', linewidth=2, label="Gradient opt")

    ax.plot(batch["batch"], startbrut["error"], marker='x', linewidth=1, linestyle = 'dashed', label="Start + Bruteforce error")
    ax.plot(batch["batch"], indipendent["error"], marker='x', linewidth=1, linestyle = 'dashed', label="Indipendent opt error")
    ax.plot(batch["batch"], gradient["error"], marker='x', linewidth=1, linestyle = 'dashed', label="Gradient opt error")

    ax.set(xlabel='Batch number (80000 events each)', ylabel='Execution time (s)',
           title='Execution time comparison')
    ax.grid()

    fig.savefig("test.png")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
