import argparse
from typing import Any, Dict


def get_cmd_line_parser() -> Dict[str, Any]:
    # Run parameters.
    parser = argparse.ArgumentParser(description="Run FedAvg on CIFAR-10")

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
        ],
    )
    parser.add_argument(
        "--num-remove",
        help="Number of classes to remove given randomly remove",
        type=int,
    )
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
        "-nc", "--num-clients", help="Number of clients", required=True, type=int
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
        "--num-aggregations",
        help="Number of server aggregations",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        help="Number of server aggregations",
        default=32,
        type=int,
    )
    return vars(parser.parse_args())
