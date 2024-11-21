# FedQUIT: On-Device Federated Unlearning via a Quasi-Competent Virtual Teacher
This is the official repository for FedQUIT: On-Device Federated Unlearning via a Quasi-Competent Virtual Teacher.

## Preliminaries
The simulation code in this repository mainly leverages TensorFlow (TF). 
Python virtual env is managed via Poetry.
See `federated_fedquit/pyproject.toml`. To reproduce our virtual env,
follow the instructions in the Environment Setup section of this readme.

The code in this repository has been tested on Ubuntu 22.04.3,
and with Python version `3.10.13`.

## Environment Setup
By default, Poetry will use the Python version in your system. 
In some settings, you might want to specify a particular version of Python 
to use inside your Poetry environment. You can do so with `pyenv`. 
Check the documentation for the different ways of installing `pyenv`,
but one easy way is using the automatic installer:

```bash
curl https://pyenv.run | bash
```
You can then install any Python version with `pyenv install <python-version>`
(e.g. `pyenv install 3.9.17`) and set that version as the one to be used. 
```bash
# cd to your federated_fedquit directory (i.e. where the `pyproject.toml` is)
pyenv install 3.10.12

pyenv local 3.10.12

# set that version for poetry
poetry env use 3.10.12
```
To build the Python environment as specified in the `pyproject.toml`, use the following commands:
```bash
# cd to your federated_fedquit directory (i.e. where the `pyproject.toml` is)

# install the base Poetry environment
poetry install

# activate the environment
poetry shell
```

## Creating Client Datasets
To exactly reproduce the label distribution we used in the paper run the following lines of code.
Note that we use the txt files in `client_data` folder.

```bash
python -m federated_fedquit.dataset_preparation dataset="cifar100" alpha=0.1 total_clients=10

python -m federated_fedquit.dataset_preparation dataset="cifar10" alpha=0.3 total_clients=10

```

## Running the Simulations
We prepared two scripts to run the experiments reported in the paper.
Note that this scripts save model checkpoints on disk (it can occupy 20-30 GB in total).
```bash
bash ./federated_fedquit/simulation_manager_cifar10.sh

bash ./federated_fedquit/simulation_manager_cifar100.sh
```

## Generating CSV Results
We prepared two scripts to generate csv files that report individual and aggregated results.
```bash
python -m federated_fedquit.generate_csv_results dataset="cifar10" alpha=0.3

python -m federated_fedquit.generate_csv_results dataset="cifar100" alpha=0.1

```