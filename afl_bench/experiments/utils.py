import argparse
import re
from functools import partial
from typing import Any, Dict, Tuple

import torch

import wandb
from afl_bench.agents.client_thread import ClientThread
from afl_bench.agents.clients.simple import Client
from afl_bench.agents.runtime_model import (
    GaussianRuntime,
    InstantRuntime,
    UniformRuntime,
)
from afl_bench.agents.server import Server
from afl_bench.agents.strategies import Strategy
from afl_bench.datasets.cifar10 import (
    load_cifar10_iid,
    load_cifar10_one_class_per_client,
    load_cifar10_randomly_remove,
    load_cifar10_restricted_subpoplulation,
    load_cifar10_sorted_partition,
)
from afl_bench.datasets.fashion_mnist import (
    load_fashion_mnist_iid,
    load_fashion_mnist_one_class_per_client,
    load_fashion_mnist_randomly_remove,
    load_fashion_mnist_restricted_subpoplulation,
    load_fashion_mnist_sorted_partition,
)
from afl_bench.models.simple_cnn import CIFAR10SimpleCNN, FashionMNISTSimpleCNN

CLIENT_TYPE_TO_MODEL = {
    "i": InstantRuntime,
    "g": GaussianRuntime,
    "u": UniformRuntime,
}

# Always use CUDA if available, otherwise use MPS if available, otherwise use CPU.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

DATASET_TO_MODEL = {
    "cifar10": ("CIFAR10SimpleCNN", lambda: CIFAR10SimpleCNN()),
    "fashion_mnist": ("FashionMNISTSimpleCNN", lambda: FashionMNISTSimpleCNN()),
}


def get_cmd_line_parser() -> Dict[str, Any]:
    # Run parameters.
    parser = argparse.ArgumentParser(description="Run FedAvg on CIFAR-10")

    # Strategy parameters.
    parser.add_argument(
        "--exp-weighting",
        help="Expontential weighting",
        type=float,
    )

    # Dataset parameters.
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset to use",
        required=True,
        choices=["cifar10", "fashion_mnist"],
    )
    parser.add_argument(
        "-dd",
        "--data-distribution",
        help="Data distribution to use",
        required=True,
        choices=[
            "iid",
            "sorted_partition",
            "one_class_per_client",
            "randomly_remove",
            "restricted_subpopulation",
        ],
    )
    parser.add_argument(
        "--num-remove",
        help="For randomly_remove, number of classes to remove given randomly remove",
        type=int,
    )
    parser.add_argument(
        "--subpopulation-size",
        help="For restricted subpopulation, size of each subpopulation (i.e. 5,10,15)",
        type=str,
    )
    parser.add_argument(
        "--subpopulation-labels",
        help="For restricted subpopulation, labels for each subpopulation (i.e. 1,2,3;4,5,6)",
        type=str,
    )

    # Client parameters.
    parser.add_argument(
        "-ci", "--client-info", help="Clients by runtime", required=True, type=str
    )

    parser.add_argument(
        "-cns",
        "--client-num-steps",
        help="Number of client steps",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-clr", "--client-lr", help="Client learning rate", default=0.001, type=float
    )

    parser.add_argument(
        "--batch-size",
        help="Number of server aggregations",
        default=32,
        type=int,
    )

    # Server parameters.
    parser.add_argument(
        "-wff", "--wait-for-full", help="Wait for full buffer", action="store_true"
    )
    parser.add_argument(
        "-bs", "--buffer-size", help="Buffer size", required=True, type=int
    )
    parser.add_argument(
        "-ms", "--ms-to-wait", help="Milliseconds to wait", required=False, type=int
    )
    parser.add_argument(
        "--num-aggregations",
        help="Number of server aggregations",
        required=True,
        type=int,
    )

    arguments = vars(parser.parse_args())

    # Parse subpopulation parameters.
    if arguments["data_distribution"] == "restricted_subpopulation":
        arguments["size_subpopulations"] = tuple(
            int(x) for x in arguments["subpopulation_size"].split(",")
        )

        arguments["labels_subpopulations"] = tuple(
            tuple(int(y) for y in x.split(","))
            for x in arguments["subpopulation_labels"].split(";")
        )
        assert len(arguments["size_subpopulations"]) == len(
            arguments["labels_subpopulations"]
        )

    # Choose dataset distribution load based on run config.
    load_functions = {
        "cifar10": {
            "iid": load_cifar10_iid,
            "one_class_per_client": load_cifar10_one_class_per_client,
            "sorted_partition": load_cifar10_sorted_partition,
            "randomly_remove": partial(
                load_cifar10_randomly_remove, arguments["num_remove"]
            ),
            "restricted_subpopulation": partial(
                load_cifar10_restricted_subpoplulation,
                arguments["size_subpopulations"],
                arguments["labels_subpopulations"],
            ),
        },
        "fashion_mnist": {
            "iid": load_fashion_mnist_iid,
            "one_class_per_client": load_fashion_mnist_one_class_per_client,
            "sorted_partition": load_fashion_mnist_sorted_partition,
            "randomly_remove": partial(
                load_fashion_mnist_randomly_remove, arguments["num_remove"]
            ),
            "restricted_subpopulation": partial(
                load_fashion_mnist_restricted_subpoplulation,
                arguments["size_subpopulations"],
                arguments["labels_subpopulations"],
            ),
        },
    }
    arguments["load_function"] = load_functions[arguments["dataset"]][
        arguments["data_distribution"]
    ]

    # Client info is of the form i0.0[1],g0.0/2.0[1], etc.
    # Where the first character indicates the
    raw_clients = arguments["client_info"].split(",")

    arguments["client_runtimes"] = []
    for raw_client in raw_clients:
        # Get number of clients of this type.
        match = re.search(r"\[(\d+)\]", raw_client)
        if match:
            num_of_type = int(match.group(1))
        else:
            num_of_type = 1

        # Get client type and split out client parameters.
        client_runtime_constructor = CLIENT_TYPE_TO_MODEL[raw_client[0]]
        client_params = [
            float(x) for x in raw_client[1 : raw_client.index("[")].split("/")
        ]

        for _ in range(num_of_type):
            arguments["client_runtimes"].append(
                client_runtime_constructor(*client_params)
            )

    # For specified dataset, get model name and model generator.
    arguments["model_info"] = DATASET_TO_MODEL[arguments["dataset"]]

    return arguments


def run_experiment(
    strategy: Strategy, args: Dict[str, Any], model_info: Tuple[str, callable]
):
    # Start a new wandb run to track this script
    model_name, model_generator = model_info
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="afl-bench",
        entity="afl-bench",
        name=f"{strategy.name} {args['dataset']} {args['data_distribution'] + ((' ' + str(args['num_remove'])) if args['num_remove'] is not None else '')}, client info {args['client_info']}, buffer size {args['buffer_size']}",
        # track hyperparameters and run metadata
        config={
            "architecture": model_name,
            "dataset": args["dataset"],
            "data_distribution": args["data_distribution"],
            "num_remove": args["num_remove"],
            "subpopulation_size": args["subpopulation_size"],
            "subpopulation_labels": args["subpopulation_labels"],
            "wait_for_full": args["wait_for_full"],
            "buffer_size": args["buffer_size"],
            "ms_to_wait": args["ms_to_wait"],
            "num_clients": len(args["client_runtimes"]),
            "client_runtimes": args["client_runtimes"],
            "client_lr": args["client_lr"],
            "client_num_steps": args["client_num_steps"],
            "num_aggregations": args["num_aggregations"],
            "batch_size": args["batch_size"],
            "exp_weighting": args["exp_weighting"],
            "device": DEVICE,
        },
    )

    trainloaders, testloaders, global_testloader = args["load_function"](
        run.config["num_clients"], batch_size=run.config["batch_size"]
    )

    #########################################################################################
    # NOTE: NOTHING BELOW THIS LINE SHOULD BE CHANGED.                                      #
    #########################################################################################

    # Define a server where global model is a SimpleCNN and strategy is the one defined above.
    server = Server(
        model_generator().to(run.config["device"]),
        strategy,
        run.config["num_aggregations"],
        global_testloader,
        device=run.config["device"],
    )

    # Create client threads with models. Note runtime is instant
    # (meaning no simulated training delay).
    client_threads = []
    for i, runtime_model in enumerate(args["client_runtimes"]):
        client = Client(
            model_generator().to(run.config["device"]),
            trainloaders[i],
            testloaders[i],
            num_steps=run.config["client_num_steps"],
            lr=run.config["client_lr"],
            device=run.config["device"],
        )
        client_thread = ClientThread(
            client, server, runtime_model=runtime_model, client_id=i
        )
        client_threads.append(client_thread)

    # Start up server and start up all client threads.
    server.run()
    for client_thread in client_threads:
        client_thread.run()

    # Kill client threads once server stops.
    server.join()
    for client_thread in client_threads:
        client_thread.stop()

    wandb.finish()
