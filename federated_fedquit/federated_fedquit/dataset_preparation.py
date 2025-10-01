"""Handle the dataset partitioning and (optionally) complex downloads.

"""
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
import os
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_datasets as tfds
# import tfds-nightly as tfds
import numpy as np
import shutil
import sys
import ast

np.set_printoptions(threshold=sys.maxsize)

def read_fedquit_distribution(dataset, total_clients, alpha, data_folder="client_data"):
    file_path = os.path.join(data_folder, dataset, "unbalanced",
                             str(round(alpha, 1)) + "_clients" + str(total_clients) + ".txt")

    # reading the data from the file
    with open(file_path) as f:
        data = f.read()

    # reconstructing the data as a dictionary
    data_mlb = ast.literal_eval(data)

    return data_mlb

@hydra.main(config_path="conf", config_name="base", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Does everything needed to get the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    ## print parsed config
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg.dataset
    alpha = cfg.alpha
    total_clients = cfg.total_clients

    folder = "federated_datasets"

    if dataset in ["cifar100"]:
        num_classes = 100
    elif dataset in ["birds"]:
        num_classes = 200
    else:
        num_classes = 10

    # if the folder exist it is deleted and the ds partitions are re-created
    # if the folder does not exist, firstly the folder is created
    # and then the ds partitions are generated
    # exist = os.path.exists(folder)
    # if not exist:
    #     os.makedirs(folder)
    folder_path = os.path.join(folder, dataset, str(total_clients), str(round(alpha, 2)))
    exist = os.path.exists(folder_path)
    if not exist:
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path, ignore_errors=True)

    if dataset in ["cifar10"]:
        (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

    elif dataset in ["cifar100"]:
        (x_train, y_train), (_, _) = tf.keras.datasets.cifar100.load_data()

    elif dataset in ["birds"]:
        # Keep order consistent with the partition generator (no shuffling!)
        train_ds = tfds.load(
            "caltech_birds2011",
            split="train",
            shuffle_files=False,
            as_supervised=True,
            try_gcs=True,  # <— use TFDS public GCS mirror instead of Drive
        )

        # Extract labels as a 1D numpy array of ints
        y_train_list = []
        x_train_list = []
        for x, y in tfds.as_numpy(train_ds):
            # Optional: resize to a common size (comment these two lines if you want original sizes)
            x = tf.image.resize(tf.convert_to_tensor(x), size=(224, 224)).numpy()
            x_train_list.append(x)
            y_train_list.append(int(y))

        y_train = np.array(y_train_list, dtype=np.int64)  # shape: (N,)
        x_train = np.array(x_train_list, dtype=np.uint8)  # shape: (N, 224, 224, 3)

    # read the distribution of per-label examples for each client
    # from txt file
    data_mlb = read_fedquit_distribution(dataset, total_clients=total_clients, alpha=alpha)

    for client in data_mlb:
        list_extracted_all_labels = data_mlb[client]
        numpy_dataset_y = y_train[list_extracted_all_labels]
        numpy_dataset_x = x_train[list_extracted_all_labels]

        ds = tf.data.Dataset.from_tensor_slices((numpy_dataset_x, numpy_dataset_y))
        ds = ds.shuffle(buffer_size=4096)

        tf.data.Dataset.save(ds, path=os.path.join(folder_path, "train", str(client)))

    path = os.path.join(os.path.join(folder_path, "train"))

    list_of_narrays = []
    for sampled_client in range(0, total_clients):
        loaded_ds = tf.data.Dataset.load(
            path=os.path.join(path, str(sampled_client)), element_spec=None, compression=None, reader_func=None
        )

        print("[Client " + str(sampled_client) + "]")
        print("Cardinality: ", tf.data.experimental.cardinality(loaded_ds).numpy())

        def count_class(counts, batch, num_classes=num_classes):
            _, labels = batch
            for i in range(num_classes):
                cc = tf.cast(labels == i, tf.int32)
                counts[i] += tf.reduce_sum(cc)
            return counts

        initial_state = dict((i, 0) for i in range(num_classes))
        counts = loaded_ds.reduce(initial_state=initial_state, reduce_func=count_class)

        # print([(k, v.numpy()) for k, v in counts.items()])
        new_dict = {k: v.numpy() for k, v in counts.items()}
        # print(new_dict)
        res = np.array([item for item in new_dict.values()])
        # print(res)
        list_of_narrays.append(res)

    distribution = np.stack(list_of_narrays)
    print(distribution)
    # saving the distribution of per-label examples in a numpy file
    # this can be useful also to draw charts about the label distrib.
    path = os.path.join(folder_path, "distribution_train.npy")
    np.save(path, distribution)


if __name__ == "__main__":
    download_and_preprocess()