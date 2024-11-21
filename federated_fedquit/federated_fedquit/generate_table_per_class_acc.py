from tbparse import SummaryReader
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import numpy as np

from federated_fedquit.dataset import get_string_distribution, \
    load_client_datasets_from_files, normalize_img, expand_dims, \
    load_selected_client_statistics, load_label_distribution_selected_client
from federated_fedquit.mia_svc import SVC_MIA, UnLearningScore
from federated_fedquit.model import create_cnn_model
from federated_fedquit.utility import get_test_dataset, preprocess_ds_test, create_model


@hydra.main(config_path="conf", config_name="generate_tables", version_base=None)
def main(cfg: DictConfig) -> None:
    print("[Start Simulation]")
    # Print parsed config
    print(OmegaConf.to_yaml(cfg))

    #  build base config
    dataset = cfg.dataset
    alpha = cfg.alpha
    alpha_dirichlet_string = get_string_distribution(alpha)
    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    clients_to_analyse = cfg.clients_to_analyse
    last_checkpoint_retrained = cfg.last_checkpoint_retrained
    algorithm = cfg.algorithm
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    best_round = cfg.best_round
    model_string = "LeNet" if dataset in ["mnist"] else "ResNet18"
    total_classes = 10 if dataset in ["mnist", "cifar10"] else 100

    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}"
                              )

    # load test set
    # ds_test = tfds.load(
    #     'mnist',
    #     split='test',
    #     shuffle_files=True,
    #     as_supervised=True,
    # )
    #
    # ds_test = ds_test.map(
    #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_test_batched = ds_test.batch(128)
    # ds_test_batched = ds_test_batched.cache()
    # ds_test_batched = ds_test_batched.prefetch(tf.data.AUTOTUNE)

    ds_test_batched = get_test_dataset(dataset)

    results = {"original": [[] for _ in range(total_clients)],
               "retrained": [[] for _ in range(total_clients)],
               "logit": [[] for _ in range(total_clients)],
               "fixed": [[] for _ in range(total_clients)],
               "softmax": [[] for _ in range(total_clients)]}

    test_acc = {"original": [],
                "retrained": [],
                "logit": [],
                "softmax": [],
                "fixed": []}

    ua = {"original": [],
          "retrained": [],
          "logit": [],
          "softmax": [],
          "fixed": []}

    mia = {"original": [],
           "retrained": [],
           "logit": [],
           "softmax": [],
           "fixed": []}

    zrf = {"original": [],
           "retrained": [],
           "logit": [],
           "softmax": [],
           "fixed": []}

    js_div = {"original": [],
           "retrained": [],
           "logit": [],
           "softmax": [],
           "fixed": []}

    unlearning_model = create_model(dataset=dataset, total_classes=total_classes)

    # for cid in range(total_clients):
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
        # ds_retain = ds_retain.map(
        #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE).map(expand_dims)
        # ds_retain = ds_retain.batch(128, drop_remainder=False)
        # ds_retain = ds_retain.cache()
        # ds_retain = ds_retain.prefetch(tf.data.AUTOTUNE)
        ds_retain = preprocess_ds_test(ds_retain, dataset)
        ds_retain = ds_retain.batch(128, drop_remainder=False)
        ds_retain = ds_retain.cache()
        ds_retain = ds_retain.prefetch(tf.data.AUTOTUNE)

        # ds_test = tfds.load(
        #     'mnist',
        #     split='test',
        #     shuffle_files=True,
        #     as_supervised=True,
        # )
        ds_test = get_test_dataset(dataset, take_n=n)

        # ds_test = ds_test.take(n).map(
        #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        # ds_test = ds_test.batch(128)
        # ds_test = ds_test.cache()
        # ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        client_train_ds = load_client_datasets_from_files(
            selected_client=int(cid),
            dataset=dataset,
            total_clients=total_clients,
            alpha=alpha,
        )
        # client_train_ds = (
        #     client_train_ds.shuffle(512, reshuffle_each_iteration=False)
        #         .map(normalize_img)
        #         .map(expand_dims)
        #         .batch(local_batch_size, drop_remainder=False)
        # )
        # client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)
        client_train_ds = preprocess_ds_test(client_train_ds, dataset)
        client_train_ds = client_train_ds.batch(local_batch_size, drop_remainder=False)
        client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)
        forgetting_ds = client_train_ds

        retrained_results = []
        for what in ["original", "retrained", "logit", "softmax", "fixed"]:
            if what == "original":
                model_checkpoint_dir = os.path.join("model_checkpoints", config_dir,
                                                    "best",
                                                    f"R_{best_round}")
                log_dir = "./logs_" + what
                reader = SummaryReader(log_dir, extra_columns={'dir_name'})
                df = reader.tensors
                base_config = os.path.join(config_dir)
                df = df[df['dir_name'].str.contains(base_config)]
                # df_train_acc = df[
                #     df['dir_name'].str.contains("acc_train")]  # train_accuracy
                #
                # train_acc_df = df_train_acc.loc[df_train_acc["step"] == 24]
                # train_acc = train_acc_df.iloc[0]["value"]

                df_test_acc = df[df['dir_name'].str.contains("accuracy")]
                test_acc_df = df_test_acc.loc[df_test_acc["step"] == best_round]
                acc = test_acc_df.iloc[0]["value"]

                # ua[what].append(str(round(train_acc, 4)) + f" (24)")
                # ua[what].append("-" + f" (24)")

                test_acc[what].append(str(
                    round(acc * 100, 2)) + f" ({best_round})")

                # --- original model on train data
                original_model = create_model(dataset=dataset,
                                              total_classes=total_classes)
                if dataset == "mnist":
                    saved_checkpoint = tf.keras.saving.load_model(
                        model_checkpoint_dir)
                    original_weights = saved_checkpoint.get_weights()
                    original_model.set_weights(original_weights)
                else:
                    original_model.load_weights(model_checkpoint_dir)

                original_model.compile(optimizer='sgd',
                                       loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                           from_logits=True),
                                       metrics=['accuracy'])

                _, acc = original_model.evaluate(client_train_ds)
                ua[what].append(str(round(100 - acc * 100, 2)) + f" ({best_round})")

                # --- MIA Efficiency
                print("-- Original --")
                results_mia = SVC_MIA(shadow_train=ds_retain,
                                      shadow_test=ds_test,
                                      target_train=forgetting_ds,
                                      target_test=None,
                                      model=original_model)
                # print(results_mia)
                mia[what].append(str(
                    round(results_mia["confidence"] * 100, 2)) + f" ({best_round})")

                zrf_temp = UnLearningScore(original_model, unlearning_model, forgetting_ds)
                print(f"[ZRF score] {what}: {zrf_temp}")
                zrf[what].append(str(round(zrf_temp, 4)))

            elif what == "retrained":
                model_checkpoint_dir = os.path.join("model_checkpoints_retrained",
                                                    config_dir, "client" + str(cid),
                                                    "checkpoints",
                                                    f"R_{last_checkpoint_retrained}")
                log_dir = "./logs_" + what
                reader = SummaryReader(log_dir, extra_columns={'dir_name'})
                df = reader.tensors
                base_config = os.path.join(config_dir)
                df = df[df['dir_name'].str.contains(base_config)]
                df_train_acc = df[
                    df['dir_name'].str.contains("acc_train")]  # train_accuracy
                matching_client = "client" + str(cid)
                df_train_acc = df_train_acc[
                    df_train_acc['dir_name'].str.contains(matching_client)]

                # MIA
                print("-- Retrained --")
                retrained_model = create_model(dataset=dataset,
                                               total_classes=total_classes)
                if dataset == "mnist":
                    saved_checkpoint = tf.keras.saving.load_model(
                        model_checkpoint_dir)
                    retrained_weights = saved_checkpoint.get_weights()
                    retrained_model.set_weights(retrained_weights)
                else:
                    retrained_model.load_weights(model_checkpoint_dir)
                retrained_model.compile(optimizer='sgd',
                                        loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                            from_logits=True),
                                        metrics=['accuracy'])
                results_mia = SVC_MIA(shadow_train=ds_retain,
                                      shadow_test=ds_test,
                                      target_train=forgetting_ds,
                                      target_test=None,
                                      model=retrained_model)

                mia_retrained = round(results_mia["confidence"] * 100, 2)
                mia[what].append(str(mia_retrained) + f" ({last_checkpoint_retrained})")

                zrf_temp = UnLearningScore(retrained_model, unlearning_model, forgetting_ds)
                print(f"[ZRF score] {what}: {zrf_temp}")
                zrf[what].append(str(round(zrf_temp, 4)))


            else:  # resumed
                log_dir = "./logs_resumed"
                reader = SummaryReader(log_dir, extra_columns={'dir_name'})
                df = reader.tensors
                # base_config = os.path.join(config_dir, algorithm)
                base_config = os.path.join(config_dir, what)
                df = df[df['dir_name'].str.contains(base_config)]
                df_train_acc = df[
                    df['dir_name'].str.contains("acc_train")]  # train_accuracy
                df_test_acc = df[
                    df['dir_name'].str.contains("accuracy")]  # test_accuracy
                matching_client = "client" + str(cid)
                df_c_t = df_test_acc[
                    df_test_acc['dir_name'].str.contains(matching_client)]
                df_c = df_c_t[df_c_t["value"] >= to_reach_value]
                df_train_acc = df_train_acc[
                    df_train_acc['dir_name'].str.contains(matching_client)]
                if df_c.empty:
                    round_recovery_completed = 220  # never reached the retrained accuracy
                    df_temp = df_c_t.loc[
                        df_c_t["step"] == round_recovery_completed]
                    acc_recovery_completed = df_temp.iloc[0]["value"]
                else:
                    round_recovery_completed = int(
                        df_c.loc[df_c['step'].idxmin(), 'step'])
                    acc_recovery_completed = df_c.loc[df_c['step'].idxmin(), 'value']
                train_acc_df = df_train_acc.loc[
                        df_train_acc["step"] == round_recovery_completed]
                train_acc = train_acc_df.iloc[0]["value"]

                    # train_acc_round = round_recovery_completed

                print(f"-- Resumed {round_recovery_completed} --")
                model_checkpoint_resumed_dir = os.path.join("model_checkpoints_resumed",
                                                            config_dir, algorithm,
                                                            f"client{cid}",
                                                            "checkpoints",
                                                            f"R_{round_recovery_completed}")

                resumed_model = create_model(dataset=dataset,
                                             total_classes=total_classes)
                if dataset == "mnist":
                    saved_checkpoint = tf.keras.saving.load_model(
                        model_checkpoint_resumed_dir)
                    resumed_weights = saved_checkpoint.get_weights()
                    resumed_model.set_weights(resumed_weights)
                else:
                    resumed_model.load_weights(model_checkpoint_resumed_dir)

                resumed_model.compile(optimizer='sgd',
                                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                          from_logits=True),
                                      metrics=['accuracy'])
                results_mia = SVC_MIA(shadow_train=ds_retain,
                                      shadow_test=ds_test,
                                      target_train=forgetting_ds,
                                      target_test=None,
                                      model=resumed_model)

                mia_temp = round(results_mia["confidence"] * 100, 2)
                wrt_retrained = round(abs(mia_retrained - mia_temp), 2)
                mia[what].append(str(wrt_retrained) + f" ({round_recovery_completed})")

                zrf_temp = UnLearningScore(resumed_model, unlearning_model, forgetting_ds)
                print(f"[ZRF score] {what}: {zrf_temp}")
                zrf[what].append(str(round(zrf_temp, 4)))

                js_div_temp = 1 - UnLearningScore(retrained_model, resumed_model, forgetting_ds)
                print(f"[JS div score] {what}: {js_div_temp}")
                js_div[what].append(str(round(js_div_temp, 4)))

                ua_temp = 100 - train_acc * 100
                wrt_retrained = round(abs(ua_retrained - ua_temp), 2)

                ua[what].append(str(
                    wrt_retrained) + f" ({round_recovery_completed})")

                test_acc[what].append(str(
                    round(acc_recovery_completed * 100,
                          2)) + f" ({round_recovery_completed})")
                model_checkpoint_dir = os.path.join("model_checkpoints_resumed",
                                                    config_dir, what,
                                                    f"client{cid}",
                                                    "checkpoints",
                                                    f"R_{round_recovery_completed}")

            client_train_ds = load_client_datasets_from_files(
                selected_client=int(cid),
                dataset=dataset,
                total_clients=total_clients,
                alpha=alpha,
            )
            client_train_ds = preprocess_ds_test(client_train_ds, dataset)
            # client_train_ds = (
            #     client_train_ds.shuffle(512)
            #         .map(normalize_img)
            #         .map(expand_dims)
            #         #.batch(local_batch_size, drop_remainder=False)
            # )

            # saved_checkpoint = tf.keras.saving.load_model(model_checkpoint_dir)
            # loaded_weights = saved_checkpoint.get_weights()
            # simple_cnn = create_cnn_model()
            # simple_cnn.set_weights(loaded_weights)
            simple_cnn = create_model(dataset=dataset,
                                      total_classes=total_classes)
            if dataset == "mnist":
                saved_checkpoint = tf.keras.saving.load_model(
                    model_checkpoint_dir)
                loaded_weights = saved_checkpoint.get_weights()
                simple_cnn.set_weights(loaded_weights)
            else:
                simple_cnn.load_weights(model_checkpoint_dir)
            simple_cnn.compile(optimizer='sgd',
                               loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                   from_logits=True),
                               metrics=['accuracy'])

            if what == "retrained":
                _, to_reach_value = simple_cnn.evaluate(ds_test_batched)
                test_acc[what].append(str(
                    round(to_reach_value * 100, 2)) + f" ({last_checkpoint_retrained})")
                # print(df_train_acc.tail())
                train_acc_df = df_train_acc.loc[
                    df_train_acc["step"] == last_checkpoint_retrained]
                train_acc = train_acc_df.iloc[0]["value"]
                ua_retrained = 100 - train_acc * 100

                ua[what].append(str(round(ua_retrained, 2)) + f" ({last_checkpoint_retrained})")

            list_accuracies, list_losses = compute_per_class_accuracy(simple_cnn,
                                                                      client_train_ds,
                                                                      num_classes=total_classes,
                                                                      what=what,
                                                                      retrained_results=retrained_results)
            if what == "retrained":
                retrained_results.append(list_accuracies)
                retrained_results.append(list_losses)

            results[what][cid] = list_accuracies

    chart_folder = os.path.join("charts", "per_class_acc")
    argmax_list = []
    for cid in range(0, 10):
        draw_and_save_chart_train_acc(y=results, chart_folder=chart_folder,
                                      dataset=dataset,
                                      alpha_string=alpha_dirichlet_string, client=cid)

        dict_res, argmax = find_percentage_best_acc(results["logit"][cid], results["softmax"][cid], results["fixed"][cid])
        argmax_list.append(argmax)
        print(f"[Client {cid}]  {dict_res}")

    print_table_acc_per_class(results, test_acc, ua, mia, total_clients=total_clients,
                              total_classes=total_classes,
                              dataset=dataset,
                              alpha=alpha,
                              argmax=argmax_list,
                              zrf=zrf,
                              js_div=js_div)


def list_to_float(list_in):
    for i in range(0, len(list_in)):
        if list_in[i] != '-':
            list_in[i] = float(list_in[i])
        else:
            list_in[i] = -1


def list_to_string(list_in):
    for i in range(0, len(list_in)):
        if list_in[i] != -1:
            list_in[i] = str(list_in[i])
        else:
            list_in[i] = "-"


def find_percentage_best_acc(logit, softmax, fixed):
    list_to_float(logit)
    list_to_float(softmax)
    list_to_float(fixed)

    d = np.array((logit, softmax, fixed))
    argmax = np.argmin(d, axis=0)
    unique, counts = np.unique(argmax, return_counts=True)
    dict_res = dict(zip(unique, counts))
    total_columns = d.shape[1]
    for k in dict_res.keys():
        dict_res[k] = dict_res[k] / total_columns

    list_to_string(logit)
    list_to_string(softmax)
    list_to_string(fixed)

    return dict_res, argmax


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


def highlight_best(list_of_results, what, argmax):
    for i in range(len(list_of_results)):
        if what == "logit" and argmax[i] == 0:
            list_of_results[i] = f"\\textbf{{{list_of_results[i]}}}"
        if what == "softmax" and argmax[i] == 1:
            list_of_results[i] = f"\\textbf{{{list_of_results[i]}}}"
        if what == "fixed" and argmax[i] == 2:
            list_of_results[i] = f"\\textbf{{{list_of_results[i]}}}"
    return list_of_results


def print_table_acc_per_class(results_dictionary_original, test_accuracy, ua, mia,
                              total_clients,
                              total_classes, dataset, alpha, argmax,
                              zrf, js_div):
    print("-------------------------------------------")
    top_k = 10
    results_dictionary = results_dictionary_original.copy()
    table_head = [str(i) for i in range(top_k)]
    table_head.insert(0, "UA")
    table_head.insert(0, "MIA Eff.")
    table_head.insert(0, "ZRF")
    table_head.insert(0, "JS-Div.")
    table_head.insert(0, "Test Acc.")
    table_head.insert(0, "Model")
    table_head.insert(0, "Client")
    print(" & ".join(table_head) + "\\" + "\\")
    # print("\midrule")
    # table body
    # for cid in range(total_clients):
    j = 0
    for cid in range(len(ua["original"])):
        most_represented_classes = find_most_represented_classes(j,
                                                                 dataset=dataset,
                                                                 total_clients=total_clients,
                                                                 alpha=alpha,
                                                                 top_k=10)

        what = "original"
        train_acc_most_rep_classes = [results_dictionary[what][j][label] for label in
                                      most_represented_classes.tolist()]
        argmax_cid = [argmax[cid][label] for label in
                                      most_represented_classes.tolist()]
        results_dictionary[what][j] = train_acc_most_rep_classes
        print("\midrule")
        results_dictionary[what][j].insert(0, ua[what][cid])
        results_dictionary[what][j].insert(0, mia[what][cid])

        results_dictionary[what][j].insert(0, f"{zrf[what][cid]}")
        results_dictionary[what][j].insert(0, f"-")

        results_dictionary[what][j].insert(0, test_accuracy[what][cid])
        results_dictionary[what][j].insert(0, "Original")
        results_dictionary[what][j].insert(0, "\multirow{{5}}{*}{" + str(cid) + "}")
        # print("len", len(results_dictionary[what][cid]))
        # print(" & ".join(results_dictionary[what][j]) + "\\" + "\\")
        print(" & ".join(results_dictionary[what][j]) + "\\" + "\\")

        what = "retrained"
        train_acc_most_rep_classes = [results_dictionary[what][j][label] for label in
                                      most_represented_classes.tolist()]
        results_dictionary[what][j] = train_acc_most_rep_classes
        results_dictionary[what][j].insert(0, ua[what][cid])
        results_dictionary[what][j].insert(0, mia[what][cid])

        results_dictionary[what][j].insert(0, f"{zrf[what][cid]}")
        results_dictionary[what][j].insert(0, f"-")

        results_dictionary[what][j].insert(0, test_accuracy[what][cid])
        results_dictionary[what][j].insert(0, "Retrained")
        results_dictionary[what][j].insert(0, "")
        # print("len", len(results_dictionary[what][cid]))
        print(" & ".join(results_dictionary[what][j]) + "\\" + "\\")

        for what in ["logit", "softmax", "fixed"]:
            train_acc_most_rep_classes = [results_dictionary[what][j][label] for label
                                          in most_represented_classes.tolist()]
            results_dictionary[what][j] = train_acc_most_rep_classes
            results_dictionary[what][j] = highlight_best(results_dictionary[what][j],
                                                         what, argmax_cid)
            results_dictionary[what][j] = [
                f"\\textcolor{{blue}}{{ {results_dictionary[what][j][label]} }}" for label in range(0, len(results_dictionary[what][j]))]

            results_dictionary[what][j].insert(0, f"\\textcolor{{blue}}{{ {ua[what][cid]} }}")
            results_dictionary[what][j].insert(0, f"\\textcolor{{blue}}{{ {mia[what][cid]} }}")

            results_dictionary[what][j].insert(0, f"{zrf[what][cid]}")
            results_dictionary[what][j].insert(0, f"{js_div[what][cid]}")


            results_dictionary[what][j].insert(0, test_accuracy[what][cid])
            results_dictionary[what][j].insert(0, what.capitalize())
            results_dictionary[what][j].insert(0, "")
            # print("len", len(results_dictionary[what][cid]))
            print(" & ".join(results_dictionary[what][j]) + "\\" + "\\")

        j = j + 1

    print("-------------------------------------------")


def compute_per_class_accuracy(model, ds_unbatched, num_classes, what,
                               retrained_results):
    list_accuracies = []
    list_losses = []
    for label in range(0, num_classes):
        per_class_ds = ds_unbatched.filter(lambda _, y: tf.equal(y, label))
        if len(list(per_class_ds)):  # check if not empty
            loss, acc = model.evaluate(per_class_ds.batch(128))
            if what not in ["retrained",
                            "original"]:  # absolute delta if comparing with baseline
                acc = abs(float(retrained_results[0][label]) - acc * 100)
                loss = abs(float(retrained_results[1][label]) - loss * 100)
                list_accuracies.append(str(round(acc, 2)))
                list_losses.append(str(round(loss, 4)))
            else:
                list_accuracies.append(str(round(acc * 100, 2)))
                list_losses.append(str(round(loss, 4)))
        else:
            list_accuracies.append("-")
            list_losses.append("-")

    return list_accuracies, list_losses


def draw_and_save_chart_train_acc(y, chart_folder, dataset, alpha_string, client):
    non_zero_classes = []
    results = {
        "original": [],
        "retrained": [],
        "logit": [],
        "fixed": [],
        "softmax": []}

    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(8, 6))

    # g = sns.lineplot(x=x["original"][client], y=list(range(len(x))))
    for what in y.keys():
        for i in range(len(y[what][client])):
            if y[what][client][i] != "-":
                results[what].append(
                    abs(float(y["retrained"][client][i]) - float(y[what][client][i])))
                if what == "retrained":
                    non_zero_classes.append(i)

    df = pd.DataFrame(
        {'class': non_zero_classes,
         'logit': results['logit'],
         'softmax': results['softmax'],
         'fixed': results['fixed'],
         })
    df = df.set_index("class")
    print(df)
    # g = sns.heatmap(df, cmap="Blues")
    g = sns.lineplot(df)

    g.set_ylabel('Accuracy', fontsize=18)
    g.set_xlabel('Class', fontsize=18)
    # g.get_legend().set_title(None)
    g.set_title(f"Absolute Delta in Per-class Train Accuracy (client {client})",
                fontsize=19, pad=20)

    # Save chart on disk
    filename = "per_class_train_acc.pdf"
    filepath = os.path.join(chart_folder, dataset, alpha_string, str(client))
    exist = os.path.exists(filepath)
    if not exist:
        os.makedirs(filepath)

    g.get_figure().savefig(os.path.join(filepath, filename),
                           format='pdf', bbox_inches='tight')
    return


if __name__ == "__main__":
    main()
