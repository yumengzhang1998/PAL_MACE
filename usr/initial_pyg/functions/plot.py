import matplotlib.pyplot as plt
import pandas as pd


def plot_training_csv(file_path: str, save_fig=True, show_fig=True):

    data = pd.read_csv(file_path)
    data_by_epoch = data.groupby("epoch").mean()
    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    for x in data_by_epoch.columns:
        if x == "step":
            continue
        vp = plt.plot(data_by_epoch.index.values, data_by_epoch[x].values, label=x)
        plt.scatter(
            data_by_epoch.index.max(), data_by_epoch[x].iloc[data_by_epoch.index.max()], color=vp[0].get_color(),
            label=r"{0}: {1:0.4f}".format(x, data_by_epoch[x].iloc[data_by_epoch.index.max()])
        )

    plt.title(file_path)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize='small')
    if save_fig:
        plt.savefig(str(file_path)+".png", bbox_inches='tight')
    if show_fig:
        plt.show()