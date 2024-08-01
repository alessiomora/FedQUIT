#!/usr/bin/bash
#unl_clients=10

# main_fl original
# main_fl retrained
# unlearning_routine x2
# main_fl resumed x2

#for (( i=0; i <= unl_clients; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=$i
#done
#
#python -m basics_unlearning.unlearning_routine algorithm="logit"
#python -m basics_unlearning.unlearning_routine algorithm="softmax"
#python -m basics_unlearning.unlearning_routine algorithm="fixed"
#
#for (( i=0; i <= unl_clients; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl unlearned_cid=$i algorithm="logit" resume_training=True
#done
#
#for (( i=0; i <= unl_clients; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl unlearned_cid=$i algorithm="softmax" resume_training=True
#done
#
#for (( i=0; i <= unl_clients; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl unlearned_cid=$i algorithm="fixed" resume_training=True
#done

# cifar100
#rounds_per_run=10
#total_rounds=50
#iterations=$total_rounds/$rounds_per_run
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl
#done
#


#rounds_per_run=10
#total_rounds=200
#iterations=$total_rounds/$rounds_per_run
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=3
#done
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=5
#done
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=6
#done
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=7
#done

#python -m basics_unlearning.unlearning_routine algorithm="logit"
#python -m basics_unlearning.unlearning_routine algorithm="softmax"
#python -m basics_unlearning.unlearning_routine algorithm="fixed"

#unl_clients=7
#for (( i=0; i <= unl_clients; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="logit" resume_training=True unlearned_cid=$i
#    python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="softmax" resume_training=True unlearned_cid=$i
#    python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="fixed" resume_training=True unlearned_cid=$i
#done

#rounds_per_run=10
#total_rounds=50
#iterations=$total_rounds/$rounds_per_run
#for (( i=1; i <= iterations; ++i ))
#do
#  python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="logit" resume_training=True
#done
#
#for (( i=1; i <= iterations; ++i ))
#do
#  python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="fixed" resume_training=True
#done
#
#for (( i=1; i <= iterations; ++i ))
#do
#  python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="softmax" resume_training=True
#done

#python -m basics_unlearning.main_fl unlearned_cid=1 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=1 algorithm="softmax" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=1 algorithm="fixed" resume_training=True
#
#python -m basics_unlearning.main_fl unlearned_cid=2 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=2 algorithm="softmax" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=2 algorithm="fixed" resume_training=True
#
#python -m basics_unlearning.main_fl unlearned_cid=7 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=7 algorithm="softmax" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=7 algorithm="fixed" resume_training=True

#python -m basics_unlearning.main_fl unlearned_cid=4 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=5 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=6 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=7 algorithm="logit" resume_training=True
#
#rounds_per_run=10
#total_rounds=200
#iterations=$total_rounds/$rounds_per_run
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=8
#done
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=1
#done
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=2
#done
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=0
#done
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=9
#done

#python -m basics_unlearning.main_fl unlearned_cid=1 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=3 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=8 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=9 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="logit" resume_training=True

#python -m basics_unlearning.main_fl unlearned_cid=1 algorithm="softmax" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=2 algorithm="softmax" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=8 algorithm="softmax" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=9 algorithm="softmax" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="softmax" resume_training=True

#python -m basics_unlearning.main_fl unlearned_cid=1 algorithm="fixed" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=2 algorithm="fixed" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=8 algorithm="fixed" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=9 algorithm="fixed" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="fixed" resume_training=True


# ---------------------------------------------------------------------------------------------
# baby - rocket
#rounds_per_run=10
#total_rounds=300
#iterations=$total_rounds/$rounds_per_run
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl
#done
#
#for (( i=1; i <= iterations; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl retraining=True
#done

#python -m basics_unlearning.unlearning_routine --multirun algorithm="ci_balanced"  learning_rate_unlearning=0.1,0.01,0.001,0.0001,0.00001 frozen_layers=0,5
#python -m basics_unlearning.unlearning_routine --multirun algorithm="ci"  learning_rate_unlearning=0.1,0.01,0.001,0.0001,0.00001 frozen_layers=0,5
#python -m basics_unlearning.unlearning_routine --multirun algorithm="logit"  learning_rate_unlearning=0.1,0.01,0.001,0.0001,0.00001 frozen_layers=0,5
#python -m basics_unlearning.unlearning_routine --multirun algorithm="softmax"  learning_rate_unlearning=0.1,0.01,0.001,0.0001,0.00001 frozen_layers=0,5
#python -m basics_unlearning.unlearning_routine --multirun algorithm="fixed"  learning_rate_unlearning=0.1,0.01,0.001,0.0001,0.00001 frozen_layers=0,5

# best performing
#python -m basics_unlearning.unlearning_routine --multirun algorithm="fixed"  learning_rate_unlearning=0.01 frozen_layers=0
#python -m basics_unlearning.unlearning_routine --multirun algorithm="softmax"  learning_rate_unlearning=0.01 frozen_layers=0
#python -m basics_unlearning.unlearning_routine --multirun algorithm="logit"  learning_rate_unlearning=0.01 frozen_layers=0
#python -m basics_unlearning.unlearning_routine --multirun algorithm="ci"  learning_rate_unlearning=0.01 frozen_layers=0
#python -m basics_unlearning.unlearning_routine --multirun algorithm="ci_balanced"  learning_rate_unlearning=0.01 frozen_layers=0

# resume training
#python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="ci" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="ci_balanced" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="logit" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="softmax" resume_training=True
#python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="fixed" resume_training=True

# natural to use as comparison
#python -m basics_unlearning.unlearning_routine algorithm="natural"  learning_rate_unlearning=1.0 frozen_layers=5
#python -m basics_unlearning.main_fl unlearned_cid=0 algorithm="natural" resume_training=True

# cifar100
#rounds_per_run=10
#total_rounds=200
#iterations=$total_rounds/$rounds_per_run
#
#for (( i=1; i <= iterations; ++i ))
#do
#  echo "$i"
#  python -m basics_unlearning.main_fl
#done

#rounds_per_run=10
#total_rounds=180
#iterations=$total_rounds/$rounds_per_run
#
#unl_clients=10
#for (( j=7; j <= unl_clients; ++j ))
#do
#  for (( i=1; i <= iterations; ++i ))
#  do
#    echo "$j"
#    python -m basics_unlearning.main_fl retraining=True unlearned_cid=$j
#  done
#done


#rounds_per_run=10
#total_rounds=60
#iterations=$total_rounds/$rounds_per_run
#for (( i=1; i <= iterations; ++i ))
#do
##  echo "$j"
#  python -m basics_unlearning.main_fl retraining=True unlearned_cid=6
#done


#python -m basics_unlearning.unlearning_routine --multirun algorithm="logit"  learning_rate_unlearning=0.1,0.01,0.001,0.0001,0.00001 frozen_layers=0,5 unlearning_epochs=1,20
#python -m basics_unlearning.unlearning_routine --multirun algorithm="softmax"  learning_rate_unlearning=0.1,0.01,0.001,0.0001,0.00001 frozen_layers=0,5 unlearning_epochs=1,20
#python -m basics_unlearning.unlearning_routine --multirun algorithm="fixed"  learning_rate_unlearning=0.1,0.01,0.001,0.0001,0.00001 frozen_layers=0,5 unlearning_epochs=1,20

#python -m basics_unlearning.unlearning_routine algorithm="logit"  learning_rate_unlearning=0.00001 frozen_layers=0 unlearning_epochs=20
#python -m basics_unlearning.unlearning_routine algorithm="softmax"  learning_rate_unlearning=0.00001 frozen_layers=0 unlearning_epochs=20
#python -m basics_unlearning.unlearning_routine --multirun algorithm="fixed"  learning_rate_unlearning=0.01 frozen_layers=5,0 unlearning_epochs=1

# prova
#python -m basics_unlearning.main_fl resume_training=True unlearned_cid=0 resuming_after_unlearning.algorithm="logit" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.0001 resuming_after_unlearning.unlearning_epochs=1
#

#unl_clients=10
#for (( i=1; i <= unl_clients; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl --multirun resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="logit" resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_epochs=20
#    python -m basics_unlearning.main_fl --multirun resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="softmax"  resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_epochs=20
##    python -m basics_unlearning.main_fl --multirun resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="fixed" resuming_after_unlearning.frozen_layers=5,0 resuming_after_unlearning.unlearning_lr=0.01 resuming_after_unlearning.unlearning_epochs=1
#done
#
#unl_clients=10
#for (( i=1; i <= unl_clients; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="logit" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.unlearning_epochs=20
#    python -m basics_unlearning.main_fl resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="softmax" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.unlearning_epochs=20
#    python -m basics_unlearning.main_fl resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="fixed" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.unlearning_epochs=20
#done
#
#
#for (( i=0; i <= unl_clients; ++i ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="logit" resuming_after_unlearning.frozen_layers=5 resuming_after_unlearning.unlearning_lr=0.01 resuming_after_unlearning.unlearning_epochs=1
#    python -m basics_unlearning.main_fl resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="softmax" resuming_after_unlearning.frozen_layers=5 resuming_after_unlearning.unlearning_lr=0.01 resuming_after_unlearning.unlearning_epochs=1
#    python -m basics_unlearning.main_fl resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="fixed" resuming_after_unlearning.frozen_layers=5 resuming_after_unlearning.unlearning_lr=0.01 resuming_after_unlearning.unlearning_epochs=1
#done


# da rifare logit e softmax prima cancellare
# resume after unlearning
#/home/amora/pycharm_projects/basics_unlearning/model_checkpoints_resumed/cifar100_0.1/ResNet18_K10_C1.0_epochs1/logit/fl_0_lr1e-05_e_20
# cancellare e rifare
# per logit e softmax proprio non è stato fatto
# per fixed manca il client 0 in una config

#python -m basics_unlearning.unlearning_routine algorithm="softmax"  learning_rate_unlearning=0.00001 frozen_layers=0 unlearning_epochs=20
#unl_clients=10
#for (( i=0; i <= unl_clients; ++i ))
#do
#    echo "$i"
##    python -m basics_unlearning.main_fl --multirun resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="logit" resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_epochs=20
#    python -m basics_unlearning.main_fl resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="softmax"  resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_epochs=20
#done

#python -m basics_unlearning.main_fl --multirun resume_training=True unlearned_cid=0 resuming_after_unlearning.algorithm="fixed" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.01 resuming_after_unlearning.unlearning_epochs=1

#cifar10
#rounds_per_run=10
#total_rounds=200
#iterations=$total_rounds/$rounds_per_run
#
#for (( i=0; i < iterations; i++ ))
#do
#  python -m basics_unlearning.main_fl dataset="cifar10" retraining=False
#done
#
#
#rounds_per_run=10
#total_rounds=200
#iterations=$total_rounds/$rounds_per_run
#unl_clients=10
#for (( j=0; j < unl_clients; j++ ))
#do
#  for (( i=0; i < iterations; i++ ))
#  do
#    echo "$j"
#    python -m basics_unlearning.main_fl dataset="cifar10" retraining=True unlearned_cid=$j
#  done
#done



#python -m basics_unlearning.unlearning_routine dataset="cifar10" algorithm="logit"  learning_rate_unlearning=0.0001 frozen_layers=0 unlearning_epochs=20
#python -m basics_unlearning.unlearning_routine dataset="cifar10" algorithm="softmax"  learning_rate_unlearning=0.0001 frozen_layers=0 unlearning_epochs=20
#python -m basics_unlearning.unlearning_routine dataset="cifar10" algorithm="fixed"  learning_rate_unlearning=0.0001 frozen_layers=0 unlearning_epochs=20

#unl_clients=8
#for (( i=7; i < unl_clients; i++ ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl dataset="cifar10" resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="logit" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.0001 resuming_after_unlearning.unlearning_epochs=20
#    python -m basics_unlearning.main_fl dataset="cifar10" resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="softmax" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.0001 resuming_after_unlearning.unlearning_epochs=20
#    python -m basics_unlearning.main_fl dataset="cifar10" resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="fixed" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.0001 resuming_after_unlearning.unlearning_epochs=20
#done
#
#python -m basics_unlearning.generate_csv_results

#python -m basics_unlearning.main_fl dataset="cifar10" resume_training=True unlearned_cid=1 resuming_after_unlearning.algorithm="logit" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.0001 resuming_after_unlearning.unlearning_epochs=20

#python -m basics_unlearning.unlearning_routine seed=0 dataset="cifar100" algorithm="softmax_zero"  learning_rate_unlearning=0.00001 frozen_layers=0 unlearning_epochs=20
#
#unl_clients=10
#for (( i=0; i < unl_clients; i++ ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl seed=0  dataset="cifar100" resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="softmax_zero" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.unlearning_epochs=20
#done


#python -m basics_unlearning.unlearning_routine seed=0 dataset="cifar100" algorithm="fixed"  learning_rate_unlearning=0.00001 frozen_layers=0 unlearning_epochs=20

#unl_clients=10
#for (( i=8; i < unl_clients; i++ ))
#do
#    echo "$i"
#    python -m basics_unlearning.main_fl seed=0  dataset="cifar100" total_rounds=4 resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="natural" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.unlearning_epochs=20
#done


python -m basics_unlearning.unlearning_routine seed=0 dataset="cifar10" algorithm="logit_min"  learning_rate_unlearning=0.0001 frozen_layers=0 unlearning_epochs=20
python -m basics_unlearning.unlearning_routine seed=0 dataset="cifar10" algorithm="softmax_zero"  learning_rate_unlearning=0.0001 frozen_layers=0 unlearning_epochs=20
python -m basics_unlearning.unlearning_routine seed=0 dataset="cifar10" algorithm="natural"  learning_rate_unlearning=0.0001 frozen_layers=0 unlearning_epochs=20

unl_clients=10
for (( i=0; i < unl_clients; i++ ))
do
    echo "$i"
    python -m basics_unlearning.main_fl seed=0  dataset="cifar10" resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="logit_min" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.0001 resuming_after_unlearning.unlearning_epochs=20
    python -m basics_unlearning.main_fl seed=0  dataset="cifar10" resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="softmax_zero" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.0001 resuming_after_unlearning.unlearning_epochs=20
    python -m basics_unlearning.main_fl seed=0  dataset="cifar10" resume_training=True unlearned_cid=$i resuming_after_unlearning.algorithm="natural" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.0001 resuming_after_unlearning.unlearning_epochs=20
done

#python -m basics_unlearning.main_fl seed=0  dataset="cifar100" resume_training=True total_rounds=2 unlearned_cid=4 resuming_after_unlearning.algorithm="logit_min" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.unlearning_epochs=20
#python -m basics_unlearning.main_fl seed=0  dataset="cifar100" resume_training=True total_rounds=2 unlearned_cid=8 resuming_after_unlearning.algorithm="logit_min" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.unlearning_epochs=20
#python -m basics_unlearning.main_fl seed=0  dataset="cifar100" resume_training=True total_rounds=2 unlearned_cid=9 resuming_after_unlearning.algorithm="logit_min" resuming_after_unlearning.frozen_layers=0 resuming_after_unlearning.unlearning_lr=0.00001 resuming_after_unlearning.unlearning_epochs=20
