
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
from federated_fedquit.utility import get_test_dataset, preprocess_ds_test, \
    create_model, compute_overlap_predictions, compute_kl_div


@hydra.main(config_path="conf", config_name="generate_tables", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #  build base config
    dataset = cfg.dataset
    alpha = cfg.alpha
    alpha_dirichlet_string = get_string_distribution(alpha)
    local_batch_size = cfg.local_batch_size
    total_clients = cfg.total_clients
    # clients_to_analyse = cfg.clients_to_analyse
    # last_checkpoint_retrained = cfg.last_checkpoint_retrained
    # algorithm = cfg.algorithm
    active_clients = cfg.active_clients
    local_epochs = cfg.local_epochs
    # best_round = cfg.best_round
    model_string = "LeNet" if dataset in ["mnist"] else "ResNet18"
    total_classes = 10 if dataset in ["mnist", "cifar10"] else 100

    ds_test_batched = get_test_dataset(dataset)

    client_train_ds_un_batched = load_client_datasets_from_files(
        selected_client=int(0),
        dataset=dataset,
        total_clients=total_clients,
        alpha=alpha,
    )
    client_train_ds_un_batched = preprocess_ds_test(client_train_ds_un_batched, dataset,
                                                    reshuffle_each_iteration=False)
    client_train_ds = client_train_ds_un_batched.batch(local_batch_size,
                                                       drop_remainder=False)
    client_train_ds = client_train_ds.prefetch(tf.data.AUTOTUNE)


    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{0}"
                              )

    model_checkpoint_dir = os.path.join("model_checkpoints_retrained", config_dir,
                                        "client0",
                                        "checkpoints",
                                        f"R_{200}")
    model_retrained_1 = create_model(dataset=dataset, total_classes=total_classes)
    model_retrained_1.load_weights(model_checkpoint_dir)
    model_retrained_1.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{1}"
                              )

    model_checkpoint_dir = os.path.join("model_checkpoints_retrained", config_dir,
                                        "client0",
                                        "checkpoints",
                                        f"R_{200}")

    model_retrained_2 = create_model(dataset=dataset, total_classes=total_classes)
    model_retrained_2.load_weights(model_checkpoint_dir)
    model_retrained_2.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

    config_dir = os.path.join(f"{dataset}_{alpha_dirichlet_string}",
                              f"{model_string}_K{total_clients}_C{active_clients}_epochs{local_epochs}_seed{0}"
                              )

    model_checkpoint_dir = os.path.join("model_checkpoints_resumed",
                                        config_dir,
                                        "logit",
                                        "fl_0_lr1e-05_e_20",
                                        "client0",
                                        "checkpoints",
                                        "R_202")

    model_unlearned = create_model(dataset=dataset, total_classes=total_classes)
    model_unlearned.load_weights(model_checkpoint_dir)
    model_unlearned.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

    # overlap
    logit_retrained_1 = model_retrained_1.predict(client_train_ds)
    logit_retrained_2 = model_retrained_2.predict(client_train_ds)
    logit_unlearned = model_unlearned.predict(client_train_ds)

    prediction_overlap_retrained = compute_overlap_predictions(
        logit_1=logit_retrained_1,
        logit_2=logit_retrained_2
    )
    prediction_overlap_u1 = compute_overlap_predictions(
        logit_1=logit_retrained_1,
        logit_2=logit_unlearned
    )
    prediction_overlap_u2 = compute_overlap_predictions(
        logit_1=logit_retrained_2,
        logit_2=logit_unlearned
    )
    print("----- Forgetting data ----- ")
    print(f"prediction_overlap_retrained: {prediction_overlap_retrained}")
    print(f"prediction_overlap_u1: {prediction_overlap_u1}")
    print(f"prediction_overlap_u2: {prediction_overlap_u2}")

    # overlap
    logit_retrained_1 = model_retrained_1.predict(ds_test_batched)
    logit_retrained_2 = model_retrained_2.predict(ds_test_batched)
    logit_unlearned = model_unlearned.predict(ds_test_batched)

    prediction_overlap_retrained = compute_overlap_predictions(
        logit_1=logit_retrained_1,
        logit_2=logit_retrained_2
    )
    prediction_overlap_u1 = compute_overlap_predictions(
        logit_1=logit_retrained_1,
        logit_2=logit_unlearned
    )
    prediction_overlap_u2 = compute_overlap_predictions(
        logit_1=logit_retrained_2,
        logit_2=logit_unlearned
    )
    print("----- Test data ----- ")
    print(f"prediction_overlap_retrained: {prediction_overlap_retrained}")
    print(f"prediction_overlap_u1: {prediction_overlap_u1}")
    print(f"prediction_overlap_u2: {prediction_overlap_u2}")


if __name__ == "__main__":
    main()