import random

from torch._utils import _accumulate
from torch.utils.data import Subset


# Helper functions.
def sort_and_partition(dataset, labels, lengths):
    """
    Sorts dataset given increasing labels and returns subsets of the dataset.
    """
    dataset_indices, _ = zip(*sorted(enumerate(labels), key=lambda x: x[1]))
    return [
        Subset(dataset, dataset_indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_list(lst, num_chunks):
    """
    Split a list into num_chunks even chunks.
    """
    chunk_size = len(lst) // num_chunks
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def one_class_partition(dataset, labels, num_classes, num_agents):
    """
    Given dataset, num_classes, labels, and num_agents, return subsets of
    the dataset such that each agent has one class and classes are split
    evenly among agents for a certain label.
    """
    assert num_agents % num_classes == 0
    num_agents_per_class = num_agents // num_classes

    agent_indices = []
    for i in range(num_classes):
        label_indices = [j for j, x in enumerate(labels) if x == i]
        agent_indices.extend(split_list(label_indices, num_agents_per_class))

    return [Subset(dataset, agent_indices[i]) for i in range(num_agents)]


def randomly_remove_labels(dataset, labels, num_classes, num_to_remove, num_agents):
    assert num_to_remove < num_classes

    # Randomly sample which agents will have which labels.
    agent_labels = []
    for _ in range(num_agents):
        agent_labels.append(
            random.sample(range(num_classes), num_classes - num_to_remove)
        )

    # Of these agents, for each label, evenly distribute to agents who need that label.
    agent_indices = [[] for _ in range(num_agents)]
    for i in range(num_classes):
        label_indices = [j for j, x in enumerate(labels) if x == i]

        random.shuffle(label_indices)
        matched_agents = [j for j in range(num_agents) if i in agent_labels[j]]

        chunk_size = len(label_indices) // len(matched_agents)

        for j, matched_agent in enumerate(matched_agents):
            agent_indices[matched_agent].extend(
                label_indices[j * chunk_size : (j + 1) * chunk_size]
            )

    return [Subset(dataset, agent_indices[i]) for i in range(num_agents)]
