import argparse
import pandas as pd
import pathlib
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statistics
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate statistics and two-way ANOVA for DQN results and create "
                                                 "graphic")
    parser.add_argument("-g", "--game", metavar="", default="Pong",
                        help='choose game for which results should be evaluated; default game is "Pong"')
    args = parser.parse_args()

    ddqn_dueling = pd.read_csv(pathlib.Path(f"results/{args.game.lower()}_dueling_ddqn.csv"))
    ddqn_single = pd.read_csv(pathlib.Path(f"results/{args.game.lower()}_single_ddqn.csv"))
    dqn_dueling = pd.read_csv(pathlib.Path(f"results/{args.game.lower()}_dueling_dqn.csv"))
    dqn_single = pd.read_csv(pathlib.Path(f"results/{args.game.lower()}_single_dqn.csv"))

    ddqn_dueling_list = ddqn_dueling["Results Evaluation"].values.tolist()
    print(f"DDQN dueling Mean: {statistics.mean(ddqn_dueling_list):.2f}, SD: {statistics.stdev(ddqn_dueling_list):.2f}")
    ddqn_single_list = ddqn_single["Results Evaluation"].values.tolist()
    print(f"DDQN single Mean: {statistics.mean(ddqn_single_list):.2f}, SD: {statistics.stdev(ddqn_single_list):.2f}")
    dqn_dueling_list = dqn_dueling["Results Evaluation"].values.tolist()
    print(f"DQN dueling Mean: {statistics.mean(dqn_dueling_list):.2f}, SD: {statistics.stdev(dqn_dueling_list):.2f}")
    dqn_single_list = dqn_single["Results Evaluation"].values.tolist()
    print(f"DQN single Mean: {statistics.mean(dqn_single_list):.2f}, SD: {statistics.stdev(dqn_single_list):.2f}")

    df = pd.concat([ddqn_dueling, ddqn_single, dqn_dueling, dqn_single], ignore_index=True)
    df = df.rename(columns={"Results Evaluation": "ResultsEvaluation"})

    is_ddqn_list = ["ddqn" if i < 60 else "dqn" for i in range(120)]
    is_dueling_list = ["dueling" if i // 30 % 2 == 0 else "single" for i in range(120)]
    df["Algorithm"] = is_ddqn_list
    df["NetworkArchitecture"] = is_dueling_list

    model = ols('ResultsEvaluation ~ C(Algorithm) + C(NetworkArchitecture) + C(Algorithm):C(NetworkArchitecture)',
                data=df).fit()
    print(sm.stats.anova_lm(model, typ=2))

    dqn_means = [statistics.mean(dqn_single_list), statistics.mean(dqn_dueling_list)]
    ddqn_means = [statistics.mean(ddqn_single_list), statistics.mean(ddqn_dueling_list)]
    dqn_sd = [statistics.stdev(dqn_single_list), statistics.stdev(dqn_dueling_list)]
    ddqn_sd = [statistics.stdev(ddqn_single_list), statistics.stdev(ddqn_dueling_list)]

    fig, ax = plt.subplots()
    ind = np.arange(2)
    width = 0.2
    ax.bar(ind, dqn_means, width, bottom=0, yerr=dqn_sd, label="DQN")
    ax.bar(ind+width, ddqn_means, width, bottom=0, yerr=ddqn_sd, label="DDQN")

    ax.set_title(f"Results {args.game}")
    ax.set_xticks(ind + width / 2, labels=["single network", "dueling network"])

    plt.subplots_adjust(right=0.8)
    ax.legend(bbox_to_anchor=(1.25, 1.025), fontsize=10)

    plt.savefig(pathlib.Path(f"figures/{args.game}.pdf"))
    plt.show()


