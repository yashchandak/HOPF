import itertools
from collections import OrderedDict

def set(obj, idx):
    # The names should be the same as argument names in parser.py
    args = OrderedDict()


    # hyper-parma search
    args['n_actions'] = [4, 8, 12]
    args['algo_name'] = ['embed_ActorCritic']
    args['true_embeddings'] = [False]
    args['load_embed'] = [True]
    args['actor_lr'] = [1e-2, 1e-3, 1e-4]
    args['critic_lr'] = [1e-2, 1e-3, 1e-4]
    args['state_lr'] = [1e-6]
    args['embed_lr'] = [1e-4]
    args['emb_lambda'] = [0, 1]
    args['emb_reg'] = [1e-1]
    args['emb_fraction'] = [0.25, 0.5, 0.75, 1]
    args['batch_size'] = [1]
    args['gamma'] = [0.99]
    args['gauss_variance'] = [1, 0.5]
    args['e_lambda'] = [0, 0.9]
    args['TIS'] = [False]

    # Fixed Hyper-params
    args['fourier_order'] = [3]
    args['reduced_action_dim'] = [2]
    args['buffer_size'] = [int(3e5)]
    args['eps'] = [0.999]
    args['save_after'] = [1000]
    args['debug'] = [False]
    args['restore'] = [False]
    args['save_model'] = [True]
    args['log_output'] = ['file_term']

    hyper_params = args.keys()
    param_values = []
    for hp_name in hyper_params:
        param_values.append(args[hp_name])

    combinations = list(itertools.product(*param_values))
    n_combinations = len(combinations)

    print('Experiment: {}/{}'.format(idx, n_combinations))
    if idx >= n_combinations:
        print("Out of bounds for Hyper param settings. Terminating...")
        exit()

    setting = combinations[idx]
    # Create command
    folder_suffix = str(idx)
    for name, value in zip(hyper_params, setting):
        setattr(obj, name, value)
        if name != 'algo_name' and len(args[name]) > 1:
            folder_suffix += "_" + str(value)
    setattr(obj, 'folder_suffix', folder_suffix)

