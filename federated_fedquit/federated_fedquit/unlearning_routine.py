"""Unlearning routine.
This scripts take in input the Original model (from disk) for a certain config,
And produces the unlearned model via a certain baseline, e.g., FedQUIT.
Then, saves this unlearned snapshot on disk. The FL training will resume from
this snapshot.
"""

import os
import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from federated_fedquit.dataset import load_client_datasets_from_files, normalize_img, \
    get_string_distribution, expand_dims, load_selected_client_statistics, \
    get_add_unlearning_label_fn, load_label_distribution_selected_client

from federated_fedquit.fedquit_training import ModelFedQuitLogitDynamic, ModelFedQuitLogitDynamicAlternative, ModelNoT
from federated_fedquit.model_projected_ga import DistanceEarlyStopping, get_distance, custom_train_loop
from federated_fedquit.utility import create_model, get_test_dataset, preprocess_ds, preprocess_ds_test, find_last_checkpoint
from federated_fedquit.baselines_utility.pga_utility import load_reference_model_for_pga
from federated_fedquit.fedquit_training import IncompetentVirtualTeacher, ModelKLDiv

tf.get_logger().setLevel(5)

@hydra.main(config_path="conf", config_name="unlearning", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg.dataset
    alpha = cfg.alpha
    algorithm = cfg.algorithm
    seed  = cfg.seed
    model = cfg.model
    alpha_dirichlet_string = get_string_distribution(alpha)
    sample_unlearning = cfg.sample_unlearning

    if dataset in ["cifar100", "birds"] and model in ["MitB0"]:
        dataset = f"{dataset}-transformer"

    if dataset in ["mnist", "cifar10"]:
        total_classes = 10
    elif dataset in ["cifar100", "cifar100-transformer"]:
        total_classes = 100
    else:       # birds (cub-200)
        total_classes = 200

    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    alpha = cfg.alpha
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    frozen_layers = cfg.frozen_layers
    epochs_unlearning = cfg.unlearning_epochs
    learning_rate_unlearning = cfg.learning_rate_unlearning
    early_stopping_threshold_pga = cfg.projected_ga.early_stopping_threshold

    model_string = model
    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                              )
    last_checkpoint_retrained = find_last_checkpoint(
        os.path.join(f"model_checkpoints", config_dir, "checkpoints"))

    model_checkpoint_dir = os.path.join(f"model_checkpoints", config_dir, "checkpoints", f"R_{last_checkpoint_retrained}")


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

    retrained_model = create_model(dataset=dataset, total_classes=total_classes)

    list_dict =[]
    list_dict_o =[]
    list_dict_r =[]
    for cid in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print(f"-------- [Client {cid}] --------")
        model_checkpoint_retrained_dir = os.path.join(f"model_checkpoints_retrained", config_dir, f"client{int(cid)}", "checkpoints", f"R_{last_checkpoint_retrained}")
        retrained_model.load_weights(model_checkpoint_retrained_dir)
        retrained_model.compile(optimizer=tf.keras.optimizers.experimental.SGD(),
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                   from_logits=True),
                               metrics=['accuracy'])

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

        client_train_ds = preprocess_ds_test(client_train_ds, dataset)
        client_train_ds = client_train_ds.batch(local_batch_size,
                                                          drop_remainder=False)
        client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)

        print("[Test Server Model]")

        server_model.set_weights(original_weights)

        # original_model.evaluate(ds_test_batched)
        original_model.evaluate(ds_test, verbose=2)

        if algorithm == "logit_min":    # Dynamic per-sample min teacher (default in paper)
            fedquit_loss = cfg.fedquit.loss
            unlearning_model = ModelFedQuitLogitDynamic(server_model, original_model, model_type=model_string, dynamic_v=True, dynamic_type="min")
        elif algorithm == "incompetent":  # Incompetent teacher
            # just equi-distributed probability over classes
            fedquit_loss = cfg.fedquit.loss
            virtual_teacher = IncompetentVirtualTeacher(num_classes=total_classes)
            unlearning_model = ModelKLDiv(server_model, virtual_teacher)
        elif algorithm == "logit_f":    # Flatten teacher
            fedquit_loss = cfg.fedquit.loss
            unlearning_model = ModelFedQuitLogitDynamicAlternative(server_model, original_model, model_type=model_string, nontrue_mode="flatten")
        elif algorithm == "logit_t":    # Topk teacher
            fedquit_loss = cfg.fedquit.loss
            unlearning_model = ModelFedQuitLogitDynamicAlternative(server_model, original_model, model_type=model_string, nontrue_mode="topk", topk=5)
        elif algorithm == "logit_r":    # Logit-ladder teacher
            fedquit_loss = cfg.fedquit.loss
            unlearning_model = ModelFedQuitLogitDynamicAlternative(server_model, original_model, model_type=model_string, nontrue_mode="rankonly_anchor")
        elif algorithm == "logit_l":    # Probability-ladder teacher
            fedquit_loss = cfg.fedquit.loss
            unlearning_model = ModelFedQuitLogitDynamicAlternative(server_model, original_model, model_type=model_string, nontrue_mode="rank_prob_ladder")
        elif algorithm == "logit_d":    # Max-entropy teacher
            fedquit_loss = cfg.fedquit.loss
            unlearning_model = ModelFedQuitLogitDynamic(server_model, original_model, model_type=model_string, dynamic_v=True, dynamic_type="entropy")
        elif algorithm == "logit_v":    # Fixed v value for teacher
            fedquit_loss = cfg.fedquit.loss
            unlearning_model = ModelFedQuitLogitDynamic(server_model, original_model, model_type=model_string, dynamic_v=False, v=cfg.fedquit.v)

        elif algorithm == "not":        # NoT baseline
            unlearning_model = ModelNoT(server_model)
            # Get model weights
            layer_weights = unlearning_model.get_weights()
            # Negate the first layer's weights
            layer_weights[0] = -layer_weights[0]
            # Set the modified weights back to the model
            unlearning_model.set_weights(layer_weights)
        elif algorithm == "projected_ga":
            model_ref, unl_client_model = load_reference_model_for_pga(server_model, cid, config_dir, dataset, total_classes, total_clients)

            dist_ref_random_lst = []
            for _ in range(10):
                FLNet = create_model(dataset=dataset, total_classes=total_classes)
                dist_ref_random_lst.append(get_distance(model_ref, FLNet))

            print(
                f'Mean distance of Reference Model to random: {np.mean(dist_ref_random_lst)}')
            threshold = np.mean(dist_ref_random_lst) / 3.0
            print(f'Radius for model_ref: {threshold}')

            unlearning_model = create_model(dataset=dataset, total_classes=total_classes)
            unlearning_model.set_weights(model_ref.get_weights())
        else:
            print("Invalid algorithm selected. Exiting")
            exit()


        if algorithm not in ["projected_ga"]:
            if model in ["ResNet18"]:
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_unlearning)
            elif model in ["MitB0"]:
                optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate_unlearning)
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                restore_best_weights=True,
                patience=3)
            if algorithm in ["logit_min", "incompetent", "logit_v", "logit_f", "logit_t", "logit_r", "logit_l", "logit_d"]:
                if fedquit_loss == "kl":
                    fq_loss = tf.keras.losses.KLDivergence()
                elif fedquit_loss == "ce":
                    fq_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
                elif fedquit_loss == "mse":
                    fq_loss = tf.keras.losses.MeanSquaredError()
                elif fedquit_loss == "cs":
                    fq_loss = tf.keras.losses.CosineSimilarity()
                else:
                    print("Choose a possible value for fedquit loss")

                unlearning_model.compile(optimizer=optimizer,
                                         loss=fq_loss,
                                         metrics=['accuracy'])
        else:
            clip_grad = 5.0  # as in the original paper
            distance_threshold_early_stopping = early_stopping_threshold_pga

            if model in ["ResNet18"]:
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=learning_rate_unlearning,
                    momentum=0.9,
                    clipnorm=clip_grad)

            elif model in ["MitB0"]:
                optimizer = tf.keras.optimizers.experimental.AdamW(
                    learning_rate=learning_rate_unlearning)

            unlearning_model.compile(optimizer=optimizer,
                                     loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                         from_logits=True),
                                     metrics=['accuracy'])


        if algorithm in ["logit_min", "incompetent", "logit_v", "logit_f", "logit_t", "logit_r", "logit_l", "logit_d"]:
            unlearning_model.model.summary(show_trainable=True)
            unlearning_model.fit(client_train_ds,
                                 epochs=epochs_unlearning,
                                 callbacks=[early_stopping_callback])
        if algorithm in ["projected_ga"]:
            custom_train_loop(unlearning_model, unl_client_model = unl_client_model, epochs=epochs_unlearning, optimizer=optimizer, train_dataset=client_train_ds, threshold=threshold, distance_early_stop=distance_threshold_early_stopping, model_ref=model_ref)
        elif algorithm in ["not"]:
            print("Do nothing..")

        print("--------------------")
        client_train_ds = load_client_datasets_from_files(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )

        client_train_ds = preprocess_ds_test(client_train_ds, dataset)
        client_train_ds = client_train_ds.batch(local_batch_size,
                                                          drop_remainder=False)
        client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)

        # ---- RETAIN DATASET ----------
        first_time = True
        for i in range(total_clients):
            if i != int(cid):
                ds = load_client_datasets_from_files(
                    selected_client=i,
                    dataset=dataset,
                    total_clients=total_clients,
                    alpha=alpha,
                )
                if first_time:
                    ds_retain = ds
                    first_time = False
                else:
                    ds_retain = ds.concatenate(ds_retain)

        ds_retain = preprocess_ds_test(ds_retain, dataset)
        ds_retain = ds_retain.batch(128)
        # -------------------------------
        results_dict_retrain = {}
        results_dict_original = {}

        print("[Original - Test]")
        _, acc_o_test = original_model.evaluate(ds_test, verbose=2)
        results_dict_original["test_acc"] = acc_o_test
        print("[Original - Train]")
        _, acc_o_train = original_model.evaluate(client_train_ds, verbose=2)
        results_dict_original["train_acc"] = acc_o_train
        print("[Original - Retain]")
        _, acc_o_retain = original_model.evaluate(ds_retain, verbose=2)
        results_dict_original["retain_acc"] = acc_o_retain

        print("[Retrain - Test]")
        _, acc_r_test = retrained_model.evaluate(ds_test, verbose=2)
        results_dict_retrain["test_acc"] = acc_r_test
        print("[Retrain - Train]")
        _, acc_r_train = retrained_model.evaluate(client_train_ds, verbose=2)
        results_dict_retrain["train_acc"] = acc_r_train
        print("[Retrain - Retain]")
        _, acc_r_retain = retrained_model.evaluate(ds_retain, verbose=2)
        results_dict_retrain["retain_acc"] = acc_r_retain

        print("[Unlearned - Test]")
        unlearning_model.compile(optimizer=tf.keras.optimizers.experimental.SGD(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(
                               from_logits=True),
                           metrics=['accuracy'])
        if algorithm not in ["natural", "not"]:
            acc_u_test = unlearning_model.evaluate(ds_test, verbose=2)
        else:
            acc_u_test = unlearning_model.evaluate(ds_test, verbose=2)
        results_dict = {}
        results_dict["test_acc"] = acc_u_test

        print("[Unlearned - Train]")
        if algorithm not in ["natural", "not"]:
            acc_u_train = unlearning_model.evaluate(client_train_ds, verbose=2)
        else:
            acc_u_train = unlearning_model.evaluate(client_train_ds, verbose=2)

        results_dict["train_acc"] = acc_u_train

        print("[Unlearned - Retain]")
        if algorithm not in ["natural", "not"]:
            acc_u_retain = unlearning_model.evaluate(ds_retain, verbose=2)
        else:
            acc_u_retain = unlearning_model.evaluate(ds_retain, verbose=2)
        results_dict["retain_acc"] = acc_u_retain

        if algorithm not in ["projected_ga"]:
            name = f"{algorithm}_fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        else:
            name = f"{algorithm}_fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}_threshold_{early_stopping_threshold_pga}"
        results_dict["name"] = name
        results_dict["client"] = cid

        results_dict_original["name"] = name
        results_dict_original["client"] = cid

        results_dict_retrain["name"] = name
        results_dict_retrain["client"] = cid

        list_dict.append(results_dict)
        list_dict_o.append(results_dict_original)
        list_dict_r.append(results_dict_retrain)

        print("[Server] Saving checkpoint... ")
        if algorithm not in ["projected_ga"]:
            if algorithm in ["logit_min", "incompetent", "logit_v", "logit_f", "logit_t", "logit_r", "logit_l", "logit_d"]:
                fedquit_loss = cfg.fedquit.loss
                logit_value = cfg.fedquit.v
                unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}_v_{logit_value}_loss_{fedquit_loss}"
            else:
                unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        else:
            unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}_threshold_{early_stopping_threshold_pga}"
        model_checkpoint_dir = os.path.join(f"model_checkpoints", config_dir, algorithm,
                                            unlearning_config,
                                            f"R_{last_checkpoint_retrained}_unlearned_client_{cid}")

        if algorithm not in ["projected_ga"]:
            unlearning_model.model.save(model_checkpoint_dir)
        else:
            unlearning_model.save(model_checkpoint_dir)

    print("Unlearning recap")
    df = pd.DataFrame(list_dict)
    print(df)
    print("Original recap")
    df_o = pd.DataFrame(list_dict_o)
    print(df_o)
    print("Retrain recap")
    df_r = pd.DataFrame(list_dict_r)
    print(df_r)

    cols = ["train_acc", "retain_acc", "test_acc"]

    # Print mean and std
    def mean_std_table(dframe):
        # keep only requested columns that exist, coerce to numeric
        sub = dframe.filter(cols).apply(pd.to_numeric, errors="coerce")
        tbl = sub.agg(["mean", "std"]).T.rename(columns={"mean": "avg"})
        # nice formatting (4 decimals; change as you like)
        return tbl

    print("\nUnlearning — mean & std")
    print(mean_std_table(df).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nOriginal — mean & std")
    print(mean_std_table(df_o).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nRetrain — mean & std")
    print(mean_std_table(df_r).to_string(float_format=lambda x: f"{x:.4f}"))

    filename = f'results_unlearning_routine.csv'
    path_to_save = os.path.join("results_csv", dataset)
    exist = os.path.exists(path_to_save)
    if not exist:
        os.makedirs(path_to_save)
    df.to_csv(os.path.join(path_to_save, filename), mode='a', header=not os.path.join(path_to_save, filename))


if __name__ == "__main__":
    main()




