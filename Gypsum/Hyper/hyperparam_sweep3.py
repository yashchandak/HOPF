import itertools
from collections import OrderedDict

def set(obj, idx):
    # The names should be the same as argument names in parser.py
    args = OrderedDict()

<<<<<<< HEAD:Swarm/Hyper/hyperparam_sweep3.py
    # Hyper-parma search
    args['n_actions'] = [4, 8, 12, 16]
=======
    # hyper-parma search
    args['n_actions'] = [16]
>>>>>>> 0c53c9760cf1c2bcc9357679e2f3ed7ff93f650b:HighD/hyper/hyperparam_sweep3.py
    args['algo_name'] = ['embed_ActorCritic']
    args['true_embeddings'] = [False]
    args['load_embed'] = [False]
    args['actor_lr'] = [1e-2]
    args['critic_lr'] = [1e-2]
    args['state_lr'] = [1e-6]
    args['embed_lr'] = [1e-4]
    args['emb_lambda'] = [0]
    args['emb_reg'] = [1, 1e-1, 1e-2, 1e-3]
    args['emb_fraction'] = [1]
    args['batch_size'] = [1]
    args['gamma'] = [0.99]
    args['gauss_variance'] = [1]
    args['e_lambda'] = [0]
    args['TIS'] = [False]

    # Fixed Hyper-params
    args['fourier_order'] = [3]
    args['reduced_action_dim'] = [2]
    args['buffer_size'] = [int(3e6)]
    args['eps'] = [0.999]
    args['save_after'] = [1000]
    args['debug'] = [False]
    args['restore'] = [False]
    args['save_model'] = [True]
    args['log_output'] = ['file']


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

