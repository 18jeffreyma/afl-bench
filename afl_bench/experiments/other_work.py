def aggregation_func(global_model: ModelParams, client_updates: List[ClientUpdate]):
    """
    Define strategy and aggregation function.
    """
    # Client updates have the following nesting:
    #  - client_update[i] = (old_model, new_model, old_model_index) is the i-th update in
    #    the buffer
    # We pivot this such that we have a list of old models, new models to diff updates.
    old_models, new_models, _ = tuple(zip(*client_updates))

    # Return new model that is just the parameter wise average of the previous.
    print([name for name, _ in global_model])

    per_pair_updates = [
        [
            new_param - old_param
            for (_, old_param), (_, new_param) in zip(old_model, new_model)
        ]
        for old_model, new_model in zip(old_models, new_models)
    ]

    # For each corresponding parameter group we compute the average of the updates
    updates = [
        torch.mean(torch.stack(param_group, 0), 0)
        for param_group in zip(*per_pair_updates)
    ]

    # assert [t.shape for t in global_model] == [t.shape for t in new_model]
    return [
        (global_name, global_param.add(update))
        for (global_name, global_param), update in zip(global_model, updates)
    ]
