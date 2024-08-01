import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import pandas as pd

# Visualizing the distribution of labels on clients
from basics_unlearning.dataset import load_label_distribution

plt.style.use("seaborn-whitegrid")
chart_folder = os.path.join("charts", "label_distribution")
root_folder = os.path.join("federated_datasets")

split = "train"
heatmap = "sattler"
# heatmap = "regular"


plt.style.use("seaborn-whitegrid")

# datasets = ["mnist", "cifar100"]
datasets = ["cifar10", "cifar100"]
# table_dataset_classes = {"cifar100": 100, "birds": 200, "aircrafts": 100, "mnist": 10}
table_dataset_classes = {"cifar100": 100, "cifar10": 10}
# table_num_clients = {"cifar100": 10, "birds": 29, "aircrafts": 65, "mnist": 10}
table_num_clients = {"cifar100": 10, "cifar10": 10}
# table_num_tickpos = {"cifar100": 1, "birds": 29, "aircrafts": 5, "mnist": 1}
table_num_tickpos = {"cifar100": 1, "cifar10": 1, }



for dataset in datasets:
    print("[Dataset] "+dataset)
    num_classes = table_dataset_classes[dataset]
    num_clients = table_num_clients[dataset]
    # dirs = os.listdir(os.path.join(root_folder, dataset+"_dirichlet", str(num_clients)))
    # for d in dirs:
    for _ in range(1):
        plt.figure(figsize=(8, 6))

        smpls_loaded = load_label_distribution(0.1, dataset, total_clients=table_num_clients[dataset])

        df = pd.DataFrame({})
        for label in range(0, num_classes):
            df[str(label)] = smpls_loaded[:, label]
        df = pd.melt(df.reset_index(), id_vars=['index'], value_vars=df.columns)
        # print(df.head())
        if heatmap == 'sattler':
            g = sns.scatterplot(
                data=df, x="index", y="variable", size="value",
                sizes=(np.min(smpls_loaded)*5, np.max(smpls_loaded)/5),
            )
        else:
            df = df.pivot("index", "variable", "value")

            g = sns.heatmap(data=df, cmap="Blues",
                            vmin=np.min(smpls_loaded), vmax=np.max(smpls_loaded),
                            yticklabels=10,
                            # xticklabels=10,
                            )
        # g.set_title(dataset.capitalize(), fontsize=16, pad=80)
        g.set_ylabel('Label', fontsize=16)
        g.set_xlabel('Client', fontsize=16)
        if heatmap == 'sattler':
            xmin, xmax = 0, num_clients
            tick_pos = np.linspace(xmin, xmax, int(num_clients/table_num_tickpos[dataset])+1)
            tick_pos[-1] = tick_pos[-1]-1
            g.set_xticks(tick_pos)
            # tick_label_list = list(range(0, num_clients+1, 1))
            # print(tick_label_list)
            # g.set_xticklabels(tick_label_list)

            # y
            ymin, ymax = 0, num_classes
            tick_pos = np.linspace(ymin, ymax, 11)
            g.set_yticks(tick_pos)
            tick_label_list = list(range(-1, num_classes, int(num_classes/10)))
            tick_label_list[0] = ""
            # g.set_yticklabels(reversed(tick_label_list))
            sns.move_legend(
                g, "lower center",
                bbox_to_anchor=(.5, 1.1), ncol=7, title=None, frameon=False,
            )
            sns.despine(top=True, right=True, left=True, bottom=True)

        plt.show()
        exist = os.path.exists(chart_folder)
        if not exist:
            os.makedirs(chart_folder)
        g.get_figure().savefig(os.path.join(chart_folder, dataset + "_distrib_" + str(0.1) + "_" +heatmap+'.pdf'),
                               dpi=100, format='pdf', bbox_inches='tight')
