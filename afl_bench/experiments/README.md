# Experiments

To run experiments using an experiment file, follow these steps:

1. Install the required dependencies listed in requirements.txt
2. Prepare your dataset and split it into non-iid shards
3. Configure the hyperparameters in the `config` dictionary
4. Run the script using the command `python fedavg.py`

For example, try the following script:

```[bash]
python afl_bench/experiments/fedavg_cifar10.py -dd iid -wff -bs 10 --num-clients 10 --num-aggregations 100
python afl_bench/experiments/fedavg_cifar10.py -dd one_class_per_client -wff -bs 10 --num-clients 10 --num-aggregations 100
python afl_bench/experiments/fedavg_cifar10.py -dd sorted_partition -wff -bs 10 --num-clients 10 --num-aggregations 100
```

The script will train a federated averaging model on the specified dataset and hyperparameters.
