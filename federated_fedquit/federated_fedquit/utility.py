import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from federated_fedquit.dataset import element_norm_cifar100, normalize_img, expand_dims, \
    element_norm_cifar10, PaddedRandomCrop, element_norm_cifar100_train, \
    element_norm_cifar20, element_norm_cifar10_train
from federated_fedquit.model import create_cnn_model, create_resnet18


def draw_and_save_heatmap(np_array_data, alpha_string,
                          chart_folder=os.path.join("charts"), mode="single_class",
                          title="Accuracy After Unlearning"):
    print("---- Drawing and saving chart ----")
    plt.figure()

    g = sns.heatmap(np_array_data, annot=True, cmap="Reds", fmt='.4f')

    if mode == "all_class":
        x_label = ""
        ll = ["test", "train", "test", "train", "kl_div (o)", "kl_div (u)"]

        # Hide major tick labels
        # g.set_xticklabels('')
        g.set_xticks(np.arange(0.5, len(ll), 1).tolist())
        g.set_xticklabels(ll)


    else:
        x_label = "Class"
        # title = "Accuracy After Unlearning"

    filename = "heatmap_" + alpha_string + "_" + title.lower().replace(" ", "_") + ".pdf"
    g.set_ylabel('Client', fontsize=18)
    g.set_xlabel(x_label, fontsize=18)
    g.set_title(title, fontsize=19, pad=20)

    exist = os.path.exists(chart_folder)
    if not exist:
        os.makedirs(chart_folder)

    g.get_figure().savefig(os.path.join(chart_folder, filename),
                           format='pdf', bbox_inches='tight')
    plt.show()
    np.save(os.path.join(chart_folder, alpha_string + "_" + title.lower().replace(" ", "_")+".npy"), np_array_data)


def compute_kl_div(predictions, num_classes):
    kl = tf.keras.losses.KLDivergence()
    pred_uniform_prob = tf.fill([tf.shape(predictions)[0], num_classes], 1 / num_classes)
    kl_div = kl(tf.nn.softmax(pred_uniform_prob), tf.nn.softmax(predictions))
    return kl_div


def compute_overlap_predictions(logit_1, logit_2):
    pred_1 = tf.nn.softmax(logit_1)
    pred_2 = tf.nn.softmax(logit_2)
    argmax_1 = tf.math.argmax(pred_1, axis=1)
    argmax_2 = tf.math.argmax(pred_2, axis=1)
    # print(argmax_1)
    # print(argmax_2)
    overlap = tf.reduce_sum(
        tf.cast(tf.equal(argmax_1, argmax_2), tf.float32)) / tf.cast(tf.size(argmax_1),
                                                                     tf.float32)
    return overlap



def create_model(dataset, total_classes, norm="group"):
    if dataset in ["mnist"]:
        model = create_cnn_model()
    elif dataset in ["cifar100", "cifar10", "cifar20"]:
        model = create_resnet18(
            num_classes=total_classes,
            # input_shape=input_shape,
            norm=norm,
            seed=1,
        )
    return model


def get_test_dataset(dataset, take_n=10000):
    if dataset in ["mnist"]:
        ds_test = tfds.load(
                    'mnist',
                    split='test',
                    shuffle_files=True,
                    as_supervised=True,
                )

        ds_test = ds_test.map(
                    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        # ds_test = ds_test.map(expand_dims)

    elif dataset in ["cifar100"]:
        cifar100_tfds = tfds.load("cifar100")
        ds_test = cifar100_tfds["test"]
        ds_test = ds_test.map(element_norm_cifar100)
    elif dataset in ["cifar10"]:
        cifar10_tfds = tfds.load("cifar10")
        ds_test = cifar10_tfds["test"]
        ds_test = ds_test.map(element_norm_cifar10)
    elif dataset in ["cifar20"]:
        cifar100_tfds = tfds.load("cifar100", as_supervised=False)
        ds_test = cifar100_tfds["test"]
        ds_test = ds_test.map(element_norm_cifar20)

    ds_test = ds_test.take(take_n)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_test


def preprocess_ds_test(ds, dataset="mnist", reshuffle_each_iteration=True):
    if dataset in ["mnist"]:
        ds = ds.shuffle(2048, reshuffle_each_iteration=reshuffle_each_iteration).map(normalize_img).map(expand_dims)
    elif dataset in ["cifar100"]:
        ds = ds.shuffle(1024, reshuffle_each_iteration=reshuffle_each_iteration).map(element_norm_cifar100_train)
    elif dataset in ["cifar10"]:
        ds = ds.shuffle(1024, reshuffle_each_iteration=reshuffle_each_iteration).map(element_norm_cifar10_train)
    elif dataset in ["cifar20"]:
        ds = ds.shuffle(1024, reshuffle_each_iteration=reshuffle_each_iteration).map(element_norm_cifar20)
    return ds


def preprocess_ds(ds, dataset="mnist"):
    if dataset in ["mnist"]:
        ds = ds.shuffle(2048).map(normalize_img).map(expand_dims)
    elif dataset in ["cifar100", "cifar20"]:
        # transform images
        rotate = tf.keras.layers.RandomRotation(0.06, seed=1)
        flip = tf.keras.layers.RandomFlip(mode="horizontal", seed=1)
        crop = PaddedRandomCrop(seed=1)

        rotate_flip_crop = tf.keras.Sequential([
            rotate,
            crop,
            flip,
        ])

        def transform_data(image, img_label):
            return rotate_flip_crop(image), img_label

        if dataset in ["cifar100"]:
            ds = ds.shuffle(1024).map(element_norm_cifar100_train).map(transform_data)
        elif dataset in ["cifar10"]:
            ds = ds.shuffle(1024).map(element_norm_cifar10_train).map(transform_data)
        else:
            ds = ds.shuffle(1024).map(element_norm_cifar20).map(transform_data)
    return ds

