#!/usr/bin/bash
## CIFAR-100, ResNet-18, Non-IID
rounds_per_run=1
total_rounds=200
iterations=$total_rounds/$rounds_per_run
unl_clients=9
#
## training original model
for (( i=1; i <= iterations; ++i ))
do
    echo "$i"
    python -m federated_fedquit.main_fedquit dataset="cifar100" alpha=0.1 retraining=False model="ResNet18" learning_rate=0.1
done

## training the retrained models
for (( i=0; i <= unl_clients; ++i ))
do
  for (( j=1; j <= iterations; ++j ))
  do
    echo "$j"
    python -m federated_fedquit.main_fedquit seed=2 dataset="cifar100" alpha=0.1 retraining=True unlearned_cid=[$i] model="ResNet18" learning_rate=0.1
  done
done

## running unlearning routine for each client (logit is logit_zero in the paper, and softmax is softmax_min in the paper)
python -m federated_fedquit.unlearning_routine dataset="cifar100" alpha=0.1 algorithm="logit_min" unlearning_epochs=1 model="ResNet18" learning_rate_unlearning=1e-04 local_batch_size=32

## recovery phase
# this early stops automatically when model utility is recovered
for (( i=0; i <= unl_clients; ++i ))
  for (( j=1; j <= iterations; ++j ))
  do
      echo "$i"
      python -m federated_fedquit.main_fedquit dataset="cifar100" alpha=0.1 model="ResNet18" learning_rate=0.1 resume_training=True unlearned_cid=[$i] resuming_after_unlearning.algorithm="logit_min" resuming_after_unlearning.unlearning_lr=1e-04 resuming_after_unlearning.unlearning_epochs=1
  done
done

## CIFAR-100, ResNet-18, IID
rounds_per_run=1
total_rounds=200
iterations=$total_rounds/$rounds_per_run
unl_clients=9
#
## training original model
for (( i=1; i <= iterations; ++i ))
do
    echo "$i"
    python -m federated_fedquit.main_fedquit dataset="cifar100" alpha=-1 retraining=False model="ResNet18" learning_rate=0.1
done

## training the retrained models
for (( i=0; i <= unl_clients; ++i ))
do
  for (( j=1; j <= iterations; ++j ))
  do
    echo "$j"
    python -m federated_fedquit.main_fedquit seed=2 dataset="cifar100" alpha=-1 retraining=True unlearned_cid=[$i] model="ResNet18" learning_rate=0.1
  done
done

## running unlearning routine for each client (logit is logit_zero in the paper, and softmax is softmax_min in the paper)
python -m federated_fedquit.unlearning_routine dataset="cifar100" alpha=-1 algorithm="logit_min" unlearning_epochs=1 model="ResNet18" learning_rate_unlearning=1e-03 local_batch_size=32

## recovery phase
# this early stops automatically when model utility is recovered
for (( i=0; i <= unl_clients; ++i ))
  for (( j=1; j <= iterations; ++j ))
  do
      echo "$i"
      python -m federated_fedquit.main_fedquit dataset="cifar100" alpha=-1 model="ResNet18" learning_rate=0.1 resume_training=True unlearned_cid=[$i] resuming_after_unlearning.algorithm="logit_min" resuming_after_unlearning.unlearning_lr=1e-03 resuming_after_unlearning.unlearning_epochs=1
  done
done