import os
import tensorflow as tf
from federated_fedquit.utility import create_model, find_last_checkpoint

def load_reference_model_for_pga(global_model, unlearning_client, config_dir, dataset,
                                 total_classes, total_clients):
    last_checkpoint_retrained = find_last_checkpoint(
        os.path.join(f"model_checkpoints", config_dir,
                     "checkpoints"))
    location = os.path.join(f"model_checkpoints", config_dir,
                            f"client_models_R{last_checkpoint_retrained}",
                            f"client{unlearning_client}")
    unl_client_model = create_model(dataset=dataset, total_classes=total_classes)
    unl_client_model.load_weights(location)
    unl_client_weights = unl_client_model.get_weights()
    global_weights = global_model.get_weights()
    n = total_clients

    pga_ref_model_weights = tf.nest.map_structure(
        lambda a, b: 1 / (n - 1) * (n * a - b),
        global_weights,
        unl_client_weights)

    pga_ref_model = create_model(dataset=dataset, total_classes=total_classes)
    pga_ref_model.set_weights(pga_ref_model_weights)
    return pga_ref_model, unl_client_model