import os
import hydra
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig, OmegaConf
import matplotlib as plt
import seaborn as sns
import pandas as pd
from federated_fedquit.dataset import load_client_datasets_from_files, normalize_img, \
    get_string_distribution, expand_dims, load_selected_client_statistics, \
    get_add_unlearning_label_fn, load_label_distribution_selected_client
from federated_fedquit.main_fl import find_last_checkpoint
from federated_fedquit.model import create_cnn_model
from federated_fedquit.model_kd_div import ModelKLDiv, ModelKLDivAdaptive, \
    ModelKLDivAdaptiveSoftmax, ModelCompetentIncompetentTeacher, ModelKLDivSoftmaxZero, \
    ModelKLDivLogitMin
from federated_fedquit.utility import draw_and_save_heatmap, compute_kl_div, \
    create_model, get_test_dataset, preprocess_ds, preprocess_ds_test
import json

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
tf.get_logger().setLevel(5)


def save_dic_as_txt(filename, dic, what):
    with open(filename, 'a') as file:
        file.write(json.dumps(dic))
        file.write("\n")


class VirtualTeacher(tf.keras.Model):
    """ A virtual teacher that outputs fixed predictions, i.e.,
    1/num_classes for each class."""

    def __init__(self, num_classes, config="fixed", mean_per_class_prob=tf.zeros([])):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.mean_per_class_prob = mean_per_class_prob

    def call(self, inputs):
        x, y = inputs
        if self.config == "fixed":
            # tf.print(tf.shape(y))
            output = tf.fill([tf.shape(x)[0], self.num_classes],
                             1.0 / self.num_classes)
        elif self.config == "not-true-forgetting":
            output = tf.fill([tf.shape(x)[0], self.num_classes],
                             1.0 / (self.num_classes - 1))
            range_idx = tf.expand_dims(tf.range(0, tf.shape(y)[0]), -1)
            # tf.print(range_idx)
            # tf.print(y)
            idx = tf.concat([tf.cast(range_idx, tf.int64), tf.expand_dims(y, -1)], axis=1)
            output = tf.tensor_scatter_nd_update(output,
                                                 indices=idx,
                                                 updates=tf.zeros(tf.shape(x)[0])
                                                 )
        elif self.config == "mean_per_classs_prob":
            output = tf.gather(self.mean_per_class_prob, y)
            range_idx = tf.expand_dims(tf.range(0, tf.shape(y)[0]), -1)
            idx = tf.concat([tf.cast(range_idx, tf.int64), tf.expand_dims(y, -1)], axis=1)
            output = tf.tensor_scatter_nd_update(tf.cast(output, tf.float32),
                                                 indices=idx,
                                                 updates=tf.zeros(tf.shape(x)[0])
                                                 )
            output = tf.nn.softmax(output, axis=1)
        return output


@hydra.main(config_path="conf", config_name="unlearning", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg.dataset
    alpha = cfg.alpha
    algorithm = cfg.algorithm
    seed  = cfg.seed

    alpha_dirichlet_string = get_string_distribution(alpha)

    if dataset in ["mnist", "cifar10"]:
        total_classes = 10
    elif dataset in ["cifar100"]:
        total_classes = 100
    else:
        total_classes = 20

    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients

    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs

    frozen_layers = cfg.frozen_layers

    epochs_unlearning = cfg.unlearning_epochs
    learning_rate_unlearning = cfg.learning_rate_unlearning

    model_string = "LeNet" if dataset in ["mnist"] else "ResNet18"
    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                              )
    last_checkpoint_retrained = find_last_checkpoint(
        os.path.join("model_checkpoints", config_dir, "checkpoints"))

    model_checkpoint_dir = os.path.join("model_checkpoints", config_dir, "checkpoints", f"R_{last_checkpoint_retrained}")

    def create_dict_for_results_list(algorithm,
                                     lr,
                                     frozen_layers,
                                     original_test_acc,
                                     original_train_acc,
                                     unl_test_acc,
                                     unl_train_acc,
                                     acc_o_test_df,
                                     acc_u_test_df,
                                     acc_o_test_dr,
                                     acc_u_test_dr
                                     ):
        dict_results = {"algorithm": algorithm,
                        "lr": lr,
                        "frozen_layers": frozen_layers,
                        "original_test_acc": original_test_acc,
                        "original_train_acc": original_train_acc,
                        "unl_test_acc": unl_test_acc,
                        "unl_train_acc": unl_train_acc,
                        "original_test_df": acc_o_test_df,
                        "unlearning_test_df": acc_u_test_df,
                        "original_test_dr": acc_o_test_dr,
                        "unlearning_test_dr": acc_u_test_dr,
        }
        return dict_results
    #------------------------

    ds_test = get_test_dataset(dataset)

    server_model = create_model(dataset=dataset, total_classes=total_classes)
    if dataset in ["mnist"]:
        saved_checkpoint = tf.keras.saving.load_model(model_checkpoint_dir)
        original_weights = saved_checkpoint.get_weights()
        # server_model = create_cnn_model()
    else:
        server_model.load_weights(model_checkpoint_dir)
        original_weights = server_model.get_weights()

    # original_model = create_cnn_model()
    original_model = create_model(dataset=dataset, total_classes=total_classes)
    original_model.set_weights(original_weights)
    original_model.compile(optimizer=tf.keras.optimizers.experimental.SGD(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(
                               from_logits=True),
                           metrics=['accuracy'])
    list_dict =[]
    for cid in range(0, 10):
        print(f"-------- [Client {cid}] --------")
        # loading client u's data
        client_train_ds = load_client_datasets_from_files(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )
        client_label_distribution = load_label_distribution_selected_client(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )
        print(client_label_distribution)
        # first_element = client_train_ds.take(1)
        # print(list(first_element)[0][0])
        # print(f"[Client {cid}] Label: {list(first_element)[0][1]}")
        # client_train_ds = (
        #     client_train_ds.shuffle(512)
        #         .map(normalize_img)
        #         .map(expand_dims)
        #         .batch(local_batch_size, drop_remainder=False)
        # )
        # client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)
        client_train_ds = preprocess_ds_test(client_train_ds, dataset)
        client_train_ds = client_train_ds.batch(local_batch_size,
                                                          drop_remainder=False)
        client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)

        print("[Test Server Model]")

        server_model.set_weights(original_weights)

        # original_model.evaluate(ds_test_batched)
        original_model.evaluate(ds_test)

        # unlearning routine with kl divergence
        # unlearning_model = ModelKLDiv(server_model, virtual_teacher)
        if algorithm == "natural":
            # just to use unlearning_model.model
            # we dont train this
            unlearning_model = ModelKLDivAdaptive(server_model, original_model)
        elif algorithm == "logit":
            unlearning_model = ModelKLDivAdaptive(server_model, original_model)
        elif algorithm == "logit_min":
            unlearning_model = ModelKLDivLogitMin(server_model, original_model)
        elif algorithm == "softmax":
            unlearning_model = ModelKLDivAdaptiveSoftmax(server_model, original_model)
        elif algorithm == "softmax_zero":
            unlearning_model = ModelKLDivSoftmaxZero(server_model, original_model)
        elif algorithm in ["ci", "ci_balanced"]:
            virtual_teacher = VirtualTeacher(num_classes=total_classes, config="fixed")
            unlearning_model = ModelCompetentIncompetentTeacher(server_model, original_model, virtual_teacher)
        else:  # just equi-distributed probability over classes
            virtual_teacher = VirtualTeacher(num_classes=total_classes, config="fixed")
            unlearning_model = ModelKLDiv(server_model, virtual_teacher)

        print("---- Unlearning ----")
        if dataset in ["mnist"]:
            unlearning_model.model.conv1.trainable = False
            unlearning_model.model.conv2.trainable = False
            unlearning_model.model.dense1.trainable = False
            unlearning_model.model.dense2.trainable = True
        else:
            if frozen_layers >= 1:
                unlearning_model.model.layer0.trainable = False
            if frozen_layers >= 2:
                unlearning_model.model.layer1.trainable = False
            if frozen_layers >= 3:
                unlearning_model.model.layer2.trainable = False
            if frozen_layers >= 4:
                unlearning_model.model.layer3.trainable = False
            if frozen_layers >= 5:
                unlearning_model.model.layer4.trainable = False
                unlearning_model.model.gap.trainable = True
            unlearning_model.model.fully_connected.trainable = True

        # optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate_unlearning)  # try 0.01 for ci
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_unlearning)  # try 0.01 for ci
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            restore_best_weights=True,
            patience=3)
        unlearning_model.compile(optimizer=optimizer,
                                 loss=tf.keras.losses.KLDivergence(),
                                 metrics=['accuracy'])

        unlearning_model.model.summary(show_trainable=True)
        if algorithm in ["logit", "softmax", "fixed", "softmax_zero", "logit_min"]:
            unlearning_model.fit(client_train_ds,
                                 epochs=epochs_unlearning,
                                 callbacks=[early_stopping_callback])
        elif algorithm in ["natural"]:
            print("Do nothing..")

        elif algorithm in ["ci", "ci_balanced"]:
            # optimizer = tf.keras.optimizers.experimental.SGD(
            #     learning_rate=learning_rate_unlearning)  # try 0.01 for ci
            # early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            #     monitor='loss',
            #     restore_best_weights=True,
            #     patience=3)
            # unlearning_model.compile(optimizer=optimizer,
            #                          loss=tf.keras.losses.KLDivergence(),
            #                          metrics=['accuracy'])
            client_train_ds = load_client_datasets_from_files(
                selected_client=int(cid),
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
            )
            train_ds_cardinality = load_selected_client_statistics(
                selected_client=int(cid),
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
            )
            print(f"[Competent-Incompetent] Taking {train_ds_cardinality} samples...")
            client_train_ds = preprocess_ds_test(client_train_ds, dataset)
            client_train_ds = client_train_ds.map(
                get_add_unlearning_label_fn(dataset, 1)
            )

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

            if algorithm not in ["ci_balanced"]:
                train_ds_cardinality = int((50000 - train_ds_cardinality) *0.3)  # taking 30% as in the original paper

            ds_retain = ds_retain.shuffle(60000).take(train_ds_cardinality)
            ds_retain = preprocess_ds_test(ds_retain, dataset)
            ds_retain = ds_retain.map(
                get_add_unlearning_label_fn(dataset, 0)
            )

            ci_ds = client_train_ds.concatenate(ds_retain)
            ci_ds = ci_ds.shuffle(60000)

            ci_ds = ci_ds.batch(local_batch_size, drop_remainder=False)
            ci_ds = ci_ds.prefetch(tf.data.AUTOTUNE)

            unlearning_model.fit(ci_ds,
                                 epochs=epochs_unlearning,
                                 callbacks=[early_stopping_callback])
        print("--------------------")

        # label = 0
        #
        # for per_class_test_ds in per_class_test_dss:
        #     print(f"[Class {label}]")
        #     per_class_test_ds = per_class_test_ds.batch(128)
        #     per_class_test_ds = per_class_test_ds.cache()
        #     per_class_test_ds = per_class_test_ds.prefetch(tf.data.AUTOTUNE)
        #
        #     print("\t Original")
        #     loss_original, acc_original = original_model.evaluate(per_class_test_ds)
        #     print("\t Unlearned")
        #     loss_original, acc_unlearned = unlearning_model.evaluate(per_class_test_ds)
        #
        #     acc_original_grid[cid, label] = acc_original
        #     acc_unlearned_grid[cid, label] = acc_unlearned
        #     difference[cid, label] = acc_original - acc_unlearned
        #     label = label + 1

        client_train_ds = load_client_datasets_from_files(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )
        # client_train_ds = (
        #     client_train_ds.shuffle(512)
        #         .map(normalize_img)
        #         .map(expand_dims)
        #         .batch(local_batch_size, drop_remainder=False)
        # )
        # client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)
        client_train_ds = preprocess_ds_test(client_train_ds, dataset)
        client_train_ds = client_train_ds.batch(local_batch_size,
                                                          drop_remainder=False)
        client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)

        print("[Original - Test]")
        _, acc_o_test = original_model.evaluate(ds_test)
        # results_dict[] = acc_o_test
        print("[Original - Train]")
        _, acc_o_train = original_model.evaluate(client_train_ds)
        # results_dict[] = acc_o_train

        print("[Unlearned - Test]")
        if algorithm not in ["natural"]:
            _, acc_u_test = unlearning_model.evaluate(ds_test)
        else:
            acc_u_test = unlearning_model.evaluate(ds_test)
        results_dict = {}
        results_dict["test_acc"] = acc_u_test
        print("[Unlearned - Train]")
        if algorithm not in ["natural"]:
            _, acc_u_train = unlearning_model.evaluate(client_train_ds)
        else:
            acc_u_train = unlearning_model.evaluate(client_train_ds)
        results_dict["train_acc"] = acc_u_train
        results_dict["name"] = f"{algorithm}_fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        results_dict["client"] = cid
        list_dict.append(results_dict)
        # pred = tf.nn.softmax(original_model.predict(client_train_ds), axis=1)
        # kl_div = compute_kl_div(pred, total_classes)
        # print(f"kl_div (original): {kl_div}")
        # results_dict[] = kl_div

        # pred = tf.nn.softmax(unlearning_model.model.predict(client_train_ds), axis=1)
        # kl_div = compute_kl_div(pred, total_classes)
        # print(f"kl_div (unlearned): {kl_div}")
        # results_dict[]] = kl_div

        print("[Server] Loading checkpoint... ")
        unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        model_checkpoint_dir = os.path.join("model_checkpoints", config_dir, algorithm,
                                            unlearning_config,
                                            f"R_{last_checkpoint_retrained}_unlearned_client_{cid}")
        # model_to_save = create_cnn_model()
        # model_to_save.set_weights(unlearning_model.get_weights())
        #
        # model_to_save.save(model_checkpoint_dir)
        unlearning_model.model.save(model_checkpoint_dir)

    df = pd.DataFrame(list_dict)
    print(df)
    filename = f'results_unlearning_routine.csv'
    path_to_save = os.path.join("results_csv", dataset)
    exist = os.path.exists(path_to_save)
    if not exist:
        os.makedirs(path_to_save)
    df.to_csv(os.path.join(path_to_save, filename), mode='a', header=not os.path.join(path_to_save, filename))

        # if dataset in ["cifar20"]:
        #     cifar100_tfds = tfds.load("cifar100", as_supervised=False)
        #     ds_test = cifar100_tfds["test"]
        #     ds_test_df = ds_test.filter(lambda x: tf.equal(x["coarse_label"], 14))\
        #         .filter(lambda x: tf.equal(x["label"], 2))
        #     ds_test_df = preprocess_ds_test(ds_test_df, dataset)
        #     ds_test_dr = ds_test.filter(lambda x: tf.equal(x["coarse_label"], 14))\
        #         .filter(lambda x: tf.not_equal(x["label"], 2))
        #     ds_test_dr = preprocess_ds_test(ds_test_dr, dataset)
        #
        #
        #     ds_test_df = ds_test_df.batch(128)
        #     ds_test_df = ds_test_df.cache()
        #     ds_test_df = ds_test_df.prefetch(tf.data.AUTOTUNE)
        #     print("[Original] Test Df")
        #     if algorithm not in ["natural"]:
        #         _, acc_o_test_df = original_model.evaluate(ds_test_df)
        #     else:
        #         acc_o_test_df = original_model.evaluate(ds_test_df)
        #     print("[Unlearned] Test Df")
        #     if algorithm not in ["natural"]:
        #         _, acc_u_test_df = unlearning_model.evaluate(ds_test_df)
        #     else:
        #         acc_u_test_df = unlearning_model.evaluate(ds_test_df)
        #
        #     ds_test_dr = ds_test_dr.batch(128)
        #     ds_test_dr = ds_test_dr.cache()
        #     ds_test_dr = ds_test_dr.prefetch(tf.data.AUTOTUNE)
        #     print("[Original] Test Dr")
        #     if algorithm not in ["natural"]:
        #         _, acc_o_test_dr = original_model.evaluate(ds_test_dr)
        #     else:
        #         acc_o_test_dr = original_model.evaluate(ds_test_dr)
        #     print("[Unlearned] Test Dr")
        #     if algorithm not in ["natural"]:
        #         _, acc_u_test_dr = unlearning_model.evaluate(ds_test_dr)
        #     else:
        #         acc_u_test_dr = unlearning_model.evaluate(ds_test_dr)
        #
        #     res = create_dict_for_results_list(algorithm, learning_rate_unlearning, frozen_layers, acc_o_test, acc_o_train, acc_u_test,
        #                                        acc_u_train, acc_o_test_df, acc_u_test_df,
        #                                        acc_o_test_dr, acc_u_test_dr)
        #     save_dic_as_txt(dataset+"_results_unlearning_routine.txt", res, what=algorithm)

    # epochs_string = " E" + str(unlearning_epochs)
    # # difference_as_npa = np.asarray(difference, dtype=np.float32)
    # draw_and_save_heatmap(acc_unlearned_grid, alpha_dirichlet_string, title="Accuracy (per class)" + epochs_string)
    # draw_and_save_heatmap(acc_original_grid, alpha_dirichlet_string, title="Original Accuracy (per class)" + epochs_string)
    # draw_and_save_heatmap(difference, alpha_dirichlet_string, title="Accuracy Delta Before-After Unlearning (per class)" + epochs_string)
    # draw_and_save_heatmap(results_dict[_dirichlet_string, title="Accuracy" + epochs_string, mode="all_class")

if __name__ == "__main__":
    main()
