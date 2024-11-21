
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig, OmegaConf
import hydra

import numpy as np

from federated_fedquit.dataset import get_string_distribution, \
    load_client_datasets_from_files, normalize_img, expand_dims, \
    load_selected_client_statistics, load_label_distribution_selected_client
from federated_fedquit.mia_svc import SVC_MIA, UnLearningScore
from federated_fedquit.model import create_cnn_model
from federated_fedquit.utility import get_test_dataset, preprocess_ds_test, \
    create_model, compute_overlap_predictions, compute_kl_div


def find_last_checkpoint(dir):
    exist = os.path.exists(dir)
    if not exist:
        return -1
    else:
        filenames = os.listdir(dir)  # get all files' and folders' names in the current directory

    dirnames = []
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(dir, filename)):  # check whether the current object is a folder or not
            filename = int(filename.replace("R_", ""))
            dirnames.append(filename)
    if not dirnames:
        return -1
    last_round_in_checkpoints = max(dirnames)
    print(f"Last checkpoint found in {dir} is from round {last_round_in_checkpoints}")
    return last_round_in_checkpoints


def find_configs_in_folder(dir):
    exist = os.path.exists(dir)
    if not exist:
        return -1
    else:
        filenames = os.listdir(dir)  # get all files' and folders' names in the current directory

    dirnames = []
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(dir, filename)):  # check whether the current object is a folder or not
            dirnames.append(filename)

    if not dirnames:
        return -1

    return dirnames

def find_most_represented_classes(cid, dataset, total_clients, alpha, top_k):
    client_label_distribution = load_label_distribution_selected_client(
        selected_client=cid,
        dataset=dataset,
        total_clients=total_clients,
        alpha=alpha,
    )
    client_label_distribution = np.argsort(client_label_distribution)
    top_rep_classes = client_label_distribution[-top_k:]
    return top_rep_classes


def compute_per_class_accuracy(model, ds_unbatched, num_classes, what,
                               retrained_results=[]):
    list_accuracies = []
    list_losses = []
    for label in range(0, num_classes):
        per_class_ds = ds_unbatched.filter(lambda _, y: tf.equal(y, label))
        if len(list(per_class_ds)):  # check if not empty
            loss, acc = model.evaluate(per_class_ds.batch(128))
            if what not in ["retrained",
                            "original"]:  # absolute delta if comparing with baseline
                acc = abs(float(retrained_results[label]) - acc * 100)
                # loss = abs(float(retrained_results[1][label]) - loss * 100)
                list_accuracies.append(str(round(acc, 2)))
                list_losses.append(str(round(loss, 4)))
            else:
                list_accuracies.append(str(round(acc * 100, 2)))
                list_losses.append(str(round(loss, 4)))
        else:
            list_accuracies.append("-")
            list_losses.append("-")

    return list_accuracies, list_losses


@hydra.main(config_path="conf", config_name="generate_tables", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #  build base config
    dataset = cfg.dataset
    alpha = cfg.alpha
    alpha_dirichlet_string = get_string_distribution(alpha)
    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    # clients_to_analyse = cfg.clients_to_analyses
    # last_checkpoint_retrained = cfg.last_checkpoint_retrained
    # algorithm = cfg.algorithm
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    # best_round = cfg.best_round
    model_string = "LeNet" if dataset in ["mnist"] else "ResNet18"
    total_classes = 10 if dataset in ["mnist", "cifar10"] else 100
    rounds_recovery = 10
    seed  = cfg.seed
    filter_string = "lr0.0001"


    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                              )

    ds_test_batched = get_test_dataset(dataset)
    entry_list = []
    for cid in range(0, 10):
        print(f"---------------------------- Client {cid} ----------------------------")
        # for MIA
        n = 10000

        first_time = True
        for i in range(total_clients):
            if i != cid:
                ds = load_client_datasets_from_files(
                    selected_client=i,
                    dataset=dataset,
                    total_clients=total_clients,
                    alpha=0.1,
                )
                if first_time:
                    ds_retain = ds
                    first_time = False
                else:
                    ds_retain = ds.concatenate(ds_retain)

        ds_retain = ds_retain.shuffle(60000).take(n)

        ds_retain = preprocess_ds_test(ds_retain, dataset)
        ds_retain = ds_retain.batch(128, drop_remainder=False)
        ds_retain = ds_retain.cache()
        ds_retain = ds_retain.prefetch(tf.data.AUTOTUNE)

        ds_test = get_test_dataset(dataset, take_n=n)

        client_train_ds_un_batched = load_client_datasets_from_files(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )

        client_train_ds_un_batched = preprocess_ds_test(client_train_ds_un_batched, dataset, reshuffle_each_iteration=False)
        client_train_ds = client_train_ds_un_batched.batch(local_batch_size, drop_remainder=False)
        client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)
        forgetting_ds = client_train_ds

        most_represented_classes = find_most_represented_classes(
            cid=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=0.1,
            top_k=5)



        for what in ["retrained",  "natural", "logit_min", "softmax_zero"]:
            entry_pd = {}
            if what in ["original", "retrained"]:
                what_string = "" if what == "original" else "_retrained"
                client_dir = os.path.join("checkpoints") if what == "original" else os.path.join(f"client{cid}", "checkpoints")
                exist = os.path.exists(os.path.join("model_checkpoints" + what_string, config_dir, client_dir))
                if exist:
                    last_checkpoint_retrained = find_last_checkpoint(os.path.join("model_checkpoints"+what_string, config_dir, client_dir))

                    model_checkpoint_dir = os.path.join("model_checkpoints"+what_string, config_dir,
                                                        client_dir,
                                                        f"R_{last_checkpoint_retrained}")

                    print(f"-- {what} last saved round: {last_checkpoint_retrained} --")

                    model = create_model(dataset=dataset,
                                                  total_classes=total_classes)
                    if dataset == "mnist":
                        saved_checkpoint = tf.keras.saving.load_model(
                            model_checkpoint_dir)
                        weights = saved_checkpoint.get_weights()
                        model.set_weights(weights)
                    else:
                        model.load_weights(model_checkpoint_dir)

                    model.compile(optimizer='sgd',
                                           loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                               from_logits=True),
                                           metrics=['accuracy'])

                    # --- MIA Efficiency
                    results_mia = SVC_MIA(shadow_train=ds_retain,
                                          shadow_test=ds_test,
                                          target_train=forgetting_ds,
                                          target_test=None,
                                          model=model)

                    _, test_acc = model.evaluate(ds_test_batched)
                    test_acc = test_acc * 100
                    _, train_acc = model.evaluate(client_train_ds)
                    train_acc = train_acc * 100
                    ua = 100.0 - train_acc
                    mia = results_mia["confidence"] * 100

                    # overlap
                    logit_retrained = model.predict(client_train_ds)

                    prediction_overlap = compute_overlap_predictions(
                        logit_1=logit_retrained,
                        logit_2=logit_retrained
                    )
                    kl_div = compute_kl_div(logit_retrained, total_classes)

                    print(f"Test acc: {test_acc} -- Train acc: {train_acc} -- ua {ua} -- mia: {mia} -- overlap {prediction_overlap} -- kl_div {kl_div}")
                    entry_pd["test_acc"] = test_acc
                    entry_pd["train_acc"] = train_acc
                    entry_pd["ua"] = ua
                    entry_pd["mia"] = mia
                    entry_pd["name"] = f"{what}_client{cid}"
                    entry_pd["cid"] = cid
                    entry_pd["prediction_overlap"] = prediction_overlap.numpy()
                    entry_pd["kl_div"] = kl_div.numpy()
                    entry_pd["algorithm"] = what

                    # per-class accuarcy
                    # print("Computing per-class accuracy..")
                    # retrained_results, _ = compute_per_class_accuracy(
                    #         model,
                    #         client_train_ds_un_batched,
                    #         num_classes = total_classes,
                    #         what = what
                    #         )
                    #
                    # results = [retrained_results[label] for label in most_represented_classes.tolist()]
                    #
                    # for most_represented_class in range(0, 5):
                    #     entry_pd["class_"+str(most_represented_class)] = results[most_represented_class]

                    print(entry_pd)
                    entry_list.append(entry_pd)

            else:  # resumed
                model_checkpoint_dir = os.path.join("model_checkpoints_resumed",
                                                                config_dir,
                                                                what)

                list_configs = find_configs_in_folder(model_checkpoint_dir)

                print(list_configs)
                print(model_checkpoint_dir)
                for unlearning_config in list_configs:
                    model_checkpoint_base_dir = os.path.join(model_checkpoint_dir,
                                                             unlearning_config,
                                                             f"client{cid}",
                                                             "checkpoints")

                    if filter_string in model_checkpoint_base_dir:

                        exist = os.path.exists(model_checkpoint_base_dir)
                        if exist:
                            print(f"-- Resumed config {unlearning_config} client {cid}--")
                            for round_recovery in range(1, rounds_recovery+1):
                                entry_pd = {}
                                r = last_checkpoint_retrained + round_recovery
                                print(f"-- Resumed config {unlearning_config} client {cid} round {r}--")
                                model_checkpoint_round_dir = os.path.join(
                                    model_checkpoint_base_dir,
                                    f"R_{r}")

                                model = create_model(dataset=dataset,
                                                     total_classes=total_classes)
                                if dataset == "mnist":
                                    saved_checkpoint = tf.keras.saving.load_model(
                                        model_checkpoint_round_dir)
                                    weights = saved_checkpoint.get_weights()
                                    model.set_weights(weights)
                                else:
                                    model.load_weights(model_checkpoint_round_dir)

                                model.compile(optimizer='sgd',
                                              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                                  from_logits=True),
                                              metrics=['accuracy'])

                                # --- MIA Efficiency
                                results_mia = SVC_MIA(shadow_train=ds_retain,
                                                      shadow_test=ds_test,
                                                      target_train=forgetting_ds,
                                                      target_test=None,
                                                      model=model)

                                _, test_acc = model.evaluate(ds_test_batched)
                                test_acc = test_acc * 100
                                _, train_acc = model.evaluate(client_train_ds)
                                train_acc = train_acc * 100
                                ua = 100.0 - train_acc
                                mia = results_mia["confidence"] * 100

                                # overlap
                                logit_u = model.predict(client_train_ds)

                                prediction_overlap = compute_overlap_predictions(
                                    logit_1=logit_retrained,
                                    logit_2=logit_u
                                )

                                kl_div = compute_kl_div(logit_u, total_classes)

                                print(f"Test acc: {test_acc} -- Train acc: {train_acc} -- ua {ua} -- mia: {mia} -- overlap {prediction_overlap} -- kl_div {kl_div}")
                                entry_pd["algorithm"] = what
                                entry_pd["round_recovery"] = round_recovery
                                entry_pd["name"] = unlearning_config
                                entry_pd["test_acc"] = test_acc
                                entry_pd["train_acc"] = train_acc
                                entry_pd["ua"] = ua
                                entry_pd["mia"] = mia
                                entry_pd["cid"] = cid
                                entry_pd["prediction_overlap"] = prediction_overlap.numpy()
                                entry_pd["kl_div"] = kl_div.numpy()

                                # per-class accuarcy
                                # print("Computing per-class accuracy..")
                                # results, _ = compute_per_class_accuracy(
                                #     model,
                                #     client_train_ds_un_batched,
                                #     num_classes=total_classes,
                                #     what=what,
                                #     retrained_results=retrained_results
                                # )
                                #
                                # results = [results[label] for label in
                                #            most_represented_classes.tolist()]
                                #
                                # for most_represented_class in range(0, 5):
                                #     entry_pd["class_" + str(most_represented_class)] = results[
                                #         most_represented_class]

                                print(entry_pd)
                                entry_list.append(entry_pd)

    df = pd.DataFrame(entry_list)
    print(df)
    # re-order columns
    df = df[[ "cid", "algorithm", "name", "round_recovery", "test_acc", "train_acc", "ua", "mia",
              "prediction_overlap", "kl_div",]]
              #"class_0", "class_1", "class_2", "class_3", "class_4"]]
    filename = f'results_unlearning_after_recovery.csv'
    path_to_save = os.path.join("results_csv", dataset)
    exist = os.path.exists(path_to_save)
    if not exist:
        os.makedirs(path_to_save)
    df.to_csv(os.path.join(path_to_save, filename), mode='a', header=True)


if __name__ == "__main__":
    main()

