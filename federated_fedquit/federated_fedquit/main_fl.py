import warnings
import os
import shutil
import hydra
import numpy as np

from basics_unlearning.generate_csv_results import compute_yeom_mia
from basics_unlearning.mia_svc import SVC_MIA

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow_datasets as tfds
from omegaconf import DictConfig, OmegaConf

from federated_fedquit.dataset import (
    load_client_datasets_from_files,
    load_selected_client_statistics,
    get_string_distribution, load_selected_clients_statistics,
    normalize_img, expand_dims, element_norm_cifar100, PaddedRandomCrop,
    get_preprocess_fn
)
from federated_fedquit.utility import create_model, get_test_dataset, preprocess_ds, \
    compute_kl_div, preprocess_ds_test


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    assert tf.config.experimental.get_memory_growth(physical_devices[0])
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

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


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))
    SAVE_ROUND_CLIENTS = 200
    dataset = cfg.dataset
    alpha = cfg.alpha
    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    total_rounds = cfg.total_rounds
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    lr_decay = cfg.lr_decay
    learning_rate = cfg.learning_rate
    resume_training = cfg.resume_training  # resume training after unlearning
    retraining = cfg.retraining  # retrain baseline
    restart_training = cfg.restart_training  # restart training from checkpoint
    seed  = cfg.seed
    if dataset in ["mnist", "cifar10"]:
        total_classes = 10
    elif dataset in ["cifar100"]:
        total_classes = 100
    else:
        total_classes = 20
    cifar20_case = "rocket"

    resumed_round = 0
    unlearned_cid = cfg.unlearned_cid
    save_checkpoint = "save_all"  ## "save_last"
    first_time = True
    checkpoint_frequency = 1 if dataset in ["mnist"] else 5

    if dataset not in ["cifar20"]:
        alpha_dirichlet_string = get_string_distribution(alpha)
    else:
        alpha_dirichlet_string = cifar20_case

    # loading test dataset
    ds_test = get_test_dataset(dataset)

    # if dataset in ["cifar20"]:
    #     ds_test_df = ds_test.filter(lambda x: tf.equal(x["label"], forgetting_label))  # 100 immagini
    #     ds_test_dr = ds_test.filter(lambda x: tf.not_equal(x["label"], forgetting_label))
    #
    #     ds_test_df = ds_test_df.batch(128)
    #     ds_test_df = ds_test_df.cache()
    #     ds_test_df = ds_test_df.prefetch(tf.data.AUTOTUNE)
    #
    #     ds_test_dr = ds_test_dr.batch(128)
    #     ds_test_dr = ds_test_dr.cache()
    #     ds_test_dr = ds_test_dr.prefetch(tf.data.AUTOTUNE)

    # server model
    server_model = create_model(dataset=dataset, total_classes=total_classes)

    model_string = "LeNet" if dataset in ["mnist"] else "ResNet18"
    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}"
                              )
    model_checkpoint_base_dir = os.path.join("model_checkpoints", config_dir,
                                             "checkpoints")

    best_round = find_last_checkpoint(model_checkpoint_base_dir)

    if resume_training:
        # creating config string for resume training
        algorithm = cfg.resuming_after_unlearning.algorithm
        frozen_layers = cfg.resuming_after_unlearning.frozen_layers
        learning_rate_unlearning = cfg.resuming_after_unlearning.unlearning_lr
        epochs_unlearning = cfg.resuming_after_unlearning.unlearning_epochs
        # unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        if algorithm not in ["projected_ga"]:
            unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}"
        else:
            early_stopping_threshold_pga = cfg.resuming_after_unlearning.early_stopping_threshold
            unlearning_config = f"fl_{frozen_layers}_lr{learning_rate_unlearning}_e_{epochs_unlearning}_threshold_{early_stopping_threshold_pga}"

    if restart_training:
        if resume_training:
            print("[Server] Loading checkpoint... ")
            model_checkpoint_base_dir = os.path.join("model_checkpoints_resumed",
                                                config_dir,
                                                algorithm,
                                                unlearning_config,
                                                "client" + str(unlearned_cid),
                                                "checkpoints")
            last_round = find_last_checkpoint(model_checkpoint_base_dir)
            if last_round > 0:
                model_checkpoint_dir = os.path.join(model_checkpoint_base_dir,
                                                    f"R_{last_round}")
                server_model.load_weights(model_checkpoint_dir)
                resumed_round = last_round
            else:
                model_checkpoint_dir = os.path.join("model_checkpoints",
                                                    config_dir,
                                                    algorithm,
                                                    unlearning_config,
                                                    f"R_{best_round}_unlearned_client_{unlearned_cid}")
                server_model.load_weights(model_checkpoint_dir)
                resumed_round = best_round
        elif retraining:
            print("[Server] Loading checkpoint... ")
            model_checkpoint_base_dir = os.path.join("model_checkpoints_retrained",
                                                config_dir,
                                                "client" + str(unlearned_cid),
                                                "checkpoints")
            last_round = find_last_checkpoint(model_checkpoint_base_dir)
            if last_round > 0:
                model_checkpoint_dir = os.path.join(model_checkpoint_base_dir,
                                                    f"R_{last_round}")
                server_model.load_weights(model_checkpoint_dir)
                resumed_round = last_round
        else:  # continue original training
            print("[Server] Loading checkpoint... ")
            model_checkpoint_base_dir = os.path.join("model_checkpoints", config_dir,
                                                     "checkpoints")
            last_round = find_last_checkpoint(model_checkpoint_base_dir)
            if last_round > 0:
                model_checkpoint_dir = os.path.join(model_checkpoint_base_dir,
                                                    f"R_{last_round}")
                server_model.load_weights(model_checkpoint_dir)
                resumed_round = last_round

    server_model.summary()
    server_model.compile(optimizer='sgd',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(
                             from_logits=True),
                         metrics=['accuracy'])

    def client_update(cid, local_model, t=1, verbose=0):
        amount_of_local_examples = load_selected_client_statistics(
            int(cid),
            total_clients=total_clients,
            alpha=alpha,
            dataset=dataset,
        )

        ds_train_client = load_client_datasets_from_files(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )
        ds_train_client_unbatched = preprocess_ds(ds_train_client, dataset)
        ds_train_client = ds_train_client_unbatched.batch(local_batch_size,
                                                          drop_remainder=False)
        ds_train_client = ds_train_client.prefetch(tf.data.AUTOTUNE)

        optimizer = tf.keras.optimizers.experimental.SGD(
            learning_rate=learning_rate * (lr_decay ** (t - 1)), # lr=0.01, 0.1 (mnist, cifar)
            weight_decay=None if dataset in ["mnist"] else 1e-3)
        local_model.compile(optimizer=optimizer,
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                from_logits=True),
                            metrics=['accuracy'])
        # Local training
        print(f"[Client {cid}] Local training..")
        local_model.fit(
            ds_train_client,
            epochs=local_epochs,
            validation_data=ds_test,
            verbose=verbose
        )
        # collect per-class mean output

        per_class_mean_output = np.zeros([total_classes, total_classes])

        # for label in range(0, total_classes):
        #     # per_class_train_dss.append(ds_train_client.filter(lambda _, y: tf.equal(y, label)))
        #     per_class_ds = ds_train_client_unbatched.filter(lambda _, y: tf.equal(y, label))
        #     if len(list(per_class_ds)):  # check if not empty
        #         predictions = tf.nn.softmax(
        #             local_model.predict(per_class_ds.batch(local_batch_size, drop_remainder=False)),
        #             axis=1)
        #         per_class_mean_output[label] = tf.reduce_mean(predictions, axis=0)
        #     else:
        #         per_class_mean_output[label].fill(1/total_classes)
        # print(per_class_mean_output)

        # for projected ga
        if t == SAVE_ROUND_CLIENTS:
            location = os.path.join(model_checkpoint_dir, f"client_models_R{t}", f"client{cid}")
            print(f"[Client {cid}] Saving model checkpoint at {location}")
            exist = os.path.exists(location)
            if not exist:
                os.makedirs(location)

            local_model.save(location)

        return local_model.get_weights(), amount_of_local_examples, per_class_mean_output

    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{seed}")
    if retraining:
        log_dir = os.path.join("logs_retrained", config_dir)
        log_dir_accuracy = os.path.join(log_dir, "client"+str(unlearned_cid), "accuracy")
        log_dir_loss = os.path.join(log_dir, "client"+str(unlearned_cid), "loss")
        test_summary_writer_acc = tf.summary.create_file_writer(log_dir_accuracy)
        test_summary_writer_loss = tf.summary.create_file_writer(log_dir_loss)

        log_dir_kl_div = os.path.join(log_dir, "client"+str(unlearned_cid), "kl_train")
        log_dir_acc_train = os.path.join(log_dir, "client"+str(unlearned_cid), "acc_train")
        log_dir_loss_train = os.path.join(log_dir, "client"+str(unlearned_cid), "loss_train")
        train_summary_writer_loss = tf.summary.create_file_writer(log_dir_loss_train)
        train_summary_writer_kl_div = tf.summary.create_file_writer(log_dir_kl_div)
        train_summary_writer_acc = tf.summary.create_file_writer(log_dir_acc_train)
        model_checkpoint_dir = os.path.join("model_checkpoints_retrained", config_dir, "client"+str(unlearned_cid))

    elif resume_training:
        # log_dir = os.path.join("logs_resumed", config_dir, algorithm)
        log_dir = os.path.join("logs_resumed", config_dir, algorithm, unlearning_config)
        log_dir_accuracy = os.path.join(log_dir, "client"+str(unlearned_cid), "accuracy")
        log_dir_loss = os.path.join(log_dir, "client"+str(unlearned_cid), "loss")
        test_summary_writer_acc = tf.summary.create_file_writer(log_dir_accuracy)
        test_summary_writer_loss = tf.summary.create_file_writer(log_dir_loss)

        log_dir_kl_div = os.path.join(log_dir, "client"+str(unlearned_cid), "kl_train")
        log_dir_acc_train = os.path.join(log_dir, "client"+str(unlearned_cid), "acc_train")
        log_dir_loss_train = os.path.join(log_dir, "client"+str(unlearned_cid), "loss_train")
        train_summary_writer_loss = tf.summary.create_file_writer(log_dir_loss_train)
        train_summary_writer_kl_div = tf.summary.create_file_writer(log_dir_kl_div)
        train_summary_writer_acc = tf.summary.create_file_writer(log_dir_acc_train)

        model_checkpoint_dir = os.path.join("model_checkpoints_resumed",
                                                 config_dir,
                                                 algorithm,
                                                 unlearning_config,
                                                 "client" + str(unlearned_cid))

    else:
        model_checkpoint_dir = os.path.join("model_checkpoints", config_dir)
        log_dir = os.path.join("logs_original", config_dir)
        log_dir_accuracy = os.path.join(log_dir, "accuracy")
        log_dir_loss = os.path.join(log_dir, "loss")
        test_summary_writer_acc = tf.summary.create_file_writer(log_dir_accuracy)
        test_summary_writer_loss = tf.summary.create_file_writer(log_dir_loss)

    max_accuracy = 0

    test_loss, test_acc = server_model.evaluate(ds_test)
    # if resumed_round == 0:
        # logging global model performance before the training starts
        # with test_summary_writer_acc.as_default():
        #     tf.summary.scalar('accuracy', test_acc, step=0)
        # with test_summary_writer_loss.as_default():
        #     tf.summary.scalar('loss', test_loss, step=0)
    # if dataset in ["cifar20"]:
    #     test_loss, test_acc = server_model.evaluate(ds_test_df)
    #     with test_summary_writer_acc.as_default():
    #         tf.summary.scalar('accuracy_df', test_acc, step=0)
    #     with test_summary_writer_loss.as_default():
    #         tf.summary.scalar('loss_df', test_loss, step=0)
    #     test_loss, test_acc = server_model.evaluate(ds_test_dr)
    #     with test_summary_writer_acc.as_default():
    #         tf.summary.scalar('accuracy_dr', test_acc, step=0)
    #     with test_summary_writer_loss.as_default():
    #         tf.summary.scalar('loss_dr', test_loss, step=0)

    retrained_computed = False
    for r in range(resumed_round + 1, resumed_round + total_rounds + 1):
        delta_w_aggregated = tf.nest.map_structure(lambda a, b: a - b,
                                                   server_model.get_weights(),
                                                   server_model.get_weights())

        if resume_training or retraining:
            m = max(total_clients * active_clients, 1) - 1
        else:
            m = max(total_clients * active_clients, 1)

        client_list = list(range(total_clients))
        if resume_training or retraining:
            client_list.remove(unlearned_cid)
        print(client_list)
        sampled_clients = np.random.choice(
            np.asarray(client_list, np.int32),
            size=int(m),
            replace=False)

        print(f"[Server] Round {r} -- Selected clients: {sampled_clients}")

        selected_client_examples = load_selected_clients_statistics(
            selected_clients=sampled_clients, alpha=alpha, dataset=dataset,
            total_clients=total_clients)

        print("Total examples ", np.sum(selected_client_examples))
        print("Local examples selected clients ", selected_client_examples)
        total_examples = np.sum(selected_client_examples)
        global_weights = server_model.get_weights()
        # aggregated_mean_output = np.zeros([total_classes, total_classes], np.float32)
        for k in sampled_clients:
            client_model = server_model
            client_model.set_weights(global_weights)
            client_weights, local_samples, pc_mean_output = client_update(k,
                                                                          client_model,
                                                                          t=r)

            # FedAvg aggregation
            delta_w_local = tf.nest.map_structure(lambda a, b: a - b,
                                                  client_model.get_weights(),
                                                  global_weights,
                                                  )

            delta_w_aggregated = tf.nest.map_structure(
                lambda a, b: a + b * (local_samples / total_examples),
                delta_w_aggregated,
                delta_w_local)
            # aggregated_mean_output = aggregated_mean_output + pc_mean_output / len(
            #     sampled_clients)

        # apply the aggregated updates
        # --> sgd with 1.0 lr
        new_global_weights = tf.nest.map_structure(lambda a, b: a + b,
                                                   global_weights,
                                                   delta_w_aggregated)
        server_model.set_weights(new_global_weights)

        # logging global model performance
        test_loss, test_acc = server_model.evaluate(ds_test)
        # with test_summary_writer_acc.as_default():
        #     tf.summary.scalar('accuracy', test_acc, step=r)
        # with test_summary_writer_loss.as_default():
        #     tf.summary.scalar('loss', test_loss, step=r)
        print(f'[Server] Round {r} -- Test accuracy: {test_acc}')
        # if dataset in ["cifar20"]:
        #     test_loss, test_acc = server_model.evaluate(ds_test_df)
        #     print(f'[Server] Round {r} -- Test Df: {test_acc}')
        #     with test_summary_writer_acc.as_default():
        #         tf.summary.scalar('accuracy_df', test_acc, step=r)
        #     with test_summary_writer_loss.as_default():
        #         tf.summary.scalar('loss_df', test_loss, step=r)
        #
        #     test_loss, test_acc = server_model.evaluate(ds_test_dr)
        #     print(f'[Server] Round {r} -- Test Dr: {test_acc}')
        #     with test_summary_writer_acc.as_default():
        #         tf.summary.scalar('accuracy_dr', test_acc, step=r)
        #     with test_summary_writer_loss.as_default():
        #         tf.summary.scalar('loss_dr', test_loss, step=r)

        if save_checkpoint == "save_last":
            if r == (resumed_round + total_rounds):
                exist = os.path.exists(os.path.join(model_checkpoint_dir, "last_checkpoint"))
                if not exist:
                    os.makedirs(os.path.join(model_checkpoint_dir, "last_checkpoint"))
                else:
                    shutil.rmtree(model_checkpoint_dir, ignore_errors=True)

                server_model.save(os.path.join(model_checkpoint_dir, "last_checkpoint", f"R_{r}"))
        elif save_checkpoint == "save_all":
            print("Saving checkpoint...")
            if resume_training:  # need for all the checkpoints for the analysis
                checkpoint_frequency = 1
            if r % checkpoint_frequency == 0:
                if first_time:
                    exist = os.path.exists(os.path.join(model_checkpoint_dir, "checkpoints"))
                    if not exist:
                        os.makedirs(os.path.join(model_checkpoint_dir, "checkpoints"))
                    else:
                        if not resume_training:
                            shutil.rmtree(model_checkpoint_dir, ignore_errors=True)
                    first_time = False
                print("Saving checkpoint global model......")
                server_model.save(os.path.join(model_checkpoint_dir, "checkpoints", f"R_{r}"))

        if retraining:
            ds_train_client = load_client_datasets_from_files(
                selected_client=int(unlearned_cid),
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
            )

            ds_train_client = preprocess_ds_test(ds_train_client, dataset)
            ds_train_client = ds_train_client.batch(local_batch_size,
                                                              drop_remainder=False)
            ds_train_client = ds_train_client.prefetch(tf.data.AUTOTUNE)

            print("[Server] Train acc ")
            loss, acc = server_model.evaluate(ds_train_client)
        elif resume_training:
            ds_train_client = load_client_datasets_from_files(
                selected_client=int(unlearned_cid),
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
            )

            ds_train_client = preprocess_ds_test(ds_train_client, dataset)
            ds_train_client = ds_train_client.batch(local_batch_size,
                                                              drop_remainder=False)
            ds_train_client = ds_train_client.prefetch(tf.data.AUTOTUNE)

            # -- retrained acc
            if not retrained_computed and algorithm not in ["natural"]:
                client_dir_r = os.path.join(f"client{unlearned_cid}", "checkpoints")

                last_checkpoint_retrained = find_last_checkpoint(
                    os.path.join("model_checkpoints_retrained", config_dir, client_dir_r))

                model_checkpoint_dir_retrained = os.path.join("model_checkpoints_retrained",
                                                    config_dir,
                                                    client_dir_r,
                                                    f"R_{last_checkpoint_retrained}")

                model_retrained = create_model(dataset=dataset,
                                     total_classes=total_classes)
                model_retrained.load_weights(model_checkpoint_dir_retrained)
                model_retrained.compile(optimizer='sgd',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                  from_logits=True),
                              metrics=['accuracy'])
                print("----- Retrained model -----")
                print("Test")
                _, test_acc_retrained = model_retrained.evaluate(ds_test)
                print("Train")
                _, train_acc_retrained = model_retrained.evaluate(ds_train_client)
                retrained_computed = False
            # ----------------

            print("[Server] Train acc ")
            loss, acc = server_model.evaluate(ds_train_client)
            # with train_summary_writer_acc.as_default():
            #     tf.summary.scalar('accuracy', acc, step=r)
            # with train_summary_writer_loss.as_default():
            #     tf.summary.scalar('loss', loss, step=r)
            print("--- kl div on train data ---")
            # pred = tf.nn.softmax(server_model.predict(ds_train_client),
            #                      axis=1)
            # pred = server_model.predict(ds_train_client)
            # kl_div = compute_kl_div(pred, total_classes)
            # with train_summary_writer_kl_div.as_default():
            #     tf.summary.scalar('kl_div', kl_div, step=r)

            if algorithm not in ["natural"] and test_acc > test_acc_retrained:
                print("----- Reached test acc of retrained model -----")
                print("Evaluating MIA success rates...")
                n = 10000

                first_time_r = True
                for i in range(total_clients):
                    if i != unlearned_cid:
                        ds = load_client_datasets_from_files(
                            selected_client=i,
                            dataset=dataset,
                            total_clients=total_clients,
                            alpha=alpha,
                        )
                        if first_time_r:
                            ds_retain = ds
                            first_time_r = False
                        else:
                            ds_retain = ds.concatenate(ds_retain)

                ds_retain = ds_retain.shuffle(60000).take(n)
                ds_retain = preprocess_ds_test(ds_retain, dataset)
                ds_retain = ds_retain.batch(128, drop_remainder=False)
                ds_retain = ds_retain.cache()
                ds_retain = ds_retain.prefetch(tf.data.AUTOTUNE)

                # --- MIA Yeom et al.
                yeom_mia_retrained = compute_yeom_mia(model_retrained,
                                            train_data=ds_retain,
                                            forget_data=ds_train_client)

                # --- MIA Efficiency
                results_mia = SVC_MIA(shadow_train=ds_retain,
                                      shadow_test=ds_test,
                                      target_train=ds_train_client,
                                      target_test=None,
                                      model=model_retrained)

                mia_retrained = results_mia["confidence"] * 100

                # --- MIA Yeom et al.
                yeom_mia = compute_yeom_mia(server_model,
                                            train_data=ds_retain,
                                            forget_data=ds_train_client)

                # --- MIA Efficiency
                results_mia = SVC_MIA(shadow_train=ds_retain,
                                      shadow_test=ds_test,
                                      target_train=ds_train_client,
                                      target_test=None,
                                      model=server_model)

                mia = results_mia["confidence"] * 100
                print(f"[Retrained] yeom_mia: {yeom_mia_retrained}, song_mia: {mia_retrained}")
                print(f"[Unlearned] yeom_mia: {yeom_mia}, song_mia: {mia}")
                break


        else:  # regular training
            # saving checkpoints for the global model with best acc
            if max_accuracy < test_acc:
                max_accuracy = test_acc

                exist = os.path.exists(os.path.join(model_checkpoint_dir, "best"))
                if not exist:
                    os.makedirs(os.path.join(model_checkpoint_dir, "best"))
                else:
                    shutil.rmtree(os.path.join(model_checkpoint_dir, "best"), ignore_errors=True)

                server_model.save(os.path.join(model_checkpoint_dir, "best", f"R_{r}"))


if __name__ == "__main__":
    main()
