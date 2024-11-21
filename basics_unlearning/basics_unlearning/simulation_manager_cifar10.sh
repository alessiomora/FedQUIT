#!/usr/bin/bash
## CIFAR-10
rounds_per_run=10
total_rounds=200
iterations=$total_rounds/$rounds_per_run
unl_clients=9
#
## training original model
for (( i=1; i <= iterations; ++i ))
do
    echo "$i"
    python -m basics_unlearning.main_fl dataset="cifar10" alpha=0.3 retraining=False
done

## training the retrained models
for (( i=1; i <= iterations; ++i ))
do
  for (( i=0; i <= unl_clients; ++i ))
  do
    echo "$i"
    python -m basics_unlearning.main_fl seed=2 dataset="cifar10" alpha=0.3 retraining=True unlearned_cid=$i
  done
done

## running unlearning routine for each client (logit is logit_zero in the paper, and softmax is softmax_min in the paper)
python -m basics_unlearning.unlearning_routine dataset="cifar10" alpha=0.3 algorithm="incorrect" unlearning_epochs=20 learning_rate_unlearning=1e-05 local_batch_size=32
python -m basics_unlearning.unlearning_routine dataset="cifar10" alpha=0.3 algorithm="projected_ga" unlearning_epochs=5 learning_rate_unlearning=1e-02 projected_ga.early_stopping_threshold=6.0 local_batch_size=512
python -m basics_unlearning.unlearning_routine dataset="cifar10" alpha=0.3 algorithm="softmax" unlearning_epochs=20 learning_rate_unlearning=1e-05 local_batch_size=32
python -m basics_unlearning.unlearning_routine dataset="cifar10" alpha=0.3 algorithm="softmax_zero" unlearning_epochs=20 learning_rate_unlearning=1e-05 local_batch_size=32
python -m basics_unlearning.unlearning_routine dataset="cifar10" alpha=0.3 algorithm="logit" unlearning_epochs=20 learning_rate_unlearning=1e-05 local_batch_size=32
python -m basics_unlearning.unlearning_routine dataset="cifar10" alpha=0.3 algorithm="logit_min" unlearning_epochs=20 learning_rate_unlearning=1e-05 local_batch_size=32

for (( i=0; i <= unl_clients; ++i ))
do
  echo "$i"
  python -m basics_unlearning.main_fl --multirun seed=2  dataset="cifar10" alpha=0.3 total_rounds=20 resume_training=True unlearned_cid=0,1,2,3,4,5,6,7,8,9 resuming_after_unlearning.algorithm="logit_min" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=1e-05 resuming_after_unlearning.unlearning_epochs=20
  python -m basics_unlearning.main_fl --multirun seed=2  dataset="cifar10" alpha=0.3 total_rounds=20 resume_training=True unlearned_cid=0,1,2,3,4,5,6,7,8,9 resuming_after_unlearning.algorithm="softmax" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=1e-05 resuming_after_unlearning.unlearning_epochs=20
  python -m basics_unlearning.main_fl --multirun seed=2  dataset="cifar10" alpha=0.3 total_rounds=20 resume_training=True unlearned_cid=0,1,2,3,4,5,6,7,8,9 resuming_after_unlearning.algorithm="softmax_zero" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=1e-05 resuming_after_unlearning.unlearning_epochs=20
  python -m basics_unlearning.main_fl --multirun seed=2  dataset="cifar10" alpha=0.3 total_rounds=20 resume_training=True unlearned_cid=0,1,2,3,4,5,6,7,8,9 resuming_after_unlearning.algorithm="incorrect" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=1e-05 resuming_after_unlearning.unlearning_epochs=20
  python -m basics_unlearning.main_fl --multirun seed=2  dataset="cifar10" alpha=0.3 total_rounds=20 resume_training=True unlearned_cid=0,1,2,3,4,5,6,7,8,9 resuming_after_unlearning.algorithm="projected_ga" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=1e-02 resuming_after_unlearning.unlearning_epochs=5 resuming_after_unlearning.early_stopping_threshold=6.0
done