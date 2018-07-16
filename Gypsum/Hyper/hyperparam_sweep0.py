import sys
import itertools
import subprocess
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.tabulate_results import write_results
from src.utils.utils import *
import time
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()

# Parameters for Hyper-param sweep
parser.add_argument("--base", default=0, help="Base counter for Hyper-param search", type=int)
parser.add_argument("--inc", default=0, help="Increment counter for Hyper-param search", type=int)
parser.add_argument("--ppgpu", default=5, help="Parallel Processes per GPU", type=int)
meta_args = parser.parse_args()

n_parallel_threads = meta_args.ppgpu
idx = meta_args.base + meta_args.inc * n_parallel_threads

# args_path = '../Experiments/' + timestamp + '/args/'
# stdout_dump_path = '../Experiments/' + timestamp + '/stdout_dumps/'
args_path = '../Experiments/args/'  # TODO
stdout_dump_path = '../Experiments/stdout_dumps/'
machine = 'gypsum'
get_results_only = False

args = OrderedDict()

# The names should be the same as argument names in parser.py
args['hyper_params'] = ['dataset', 'batch_size', 'dims', 'neighbors', 'max_depth', 'lr', 'l2',
                        'drop_in', 'wce', 'percents', 'folds', 'skip_connections',
                        'dense_connections', 'propModel', 'timestamp', 'algos']

args['dataset'] = ['cora']
args['batch_size'] = [128]  # 16
args['dims'] = ['64,64,64,64,64,64,64,64,64,64']
args['neighbors'] = ['all,all,all,all']
args['max_depth'] = [1, 2]  # 1
args['lr'] = [1e-2]
args['l2'] = [0.]
args['drop_in'] = [0.]
args['wce'] = [True]
args['percents'] = [10]
args['folds'] = ['1,2,3,4,5']
args['skip_connections'] = [True]
args['dense_connections'] = [False]
args['propModel'] = ['propagation']

format = ['aggKernel', 'node_features', 'neighbor_features', 'shared_weights', 'max_outer']
args['algos'] = [
                   ['simple', 'h', '-', 0, 5],       # Node
                   ['simple', 'x', 'h', 0, 5],       # Node#
                   ['simple', 'h', '-', 0, 1],       # Node
                   ['simple', '-', 'h', 0, 1],       # Neighbor
                   ['simple', 'x', 'h', 0, 1],       # Simple
                   ['kipf', 'x', 'h', 0, 1],         # Kipf GCN
                   ['kipf', 'x', 'h', 1, 1],         # Kipf GCN
                   ['simple', 'h', 'h', 1, 1],       # Simple
                 ]

pos = args['hyper_params'].index('dataset')
args['hyper_params'][0], args['hyper_params'][pos] = args['hyper_params'][pos], args['hyper_params'][0]


if not get_results_only:

    def diff(t_a, t_b):
        t_diff = relativedelta(t_a, t_b)
        return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

    param_values = []
    this_module = sys.modules[__name__]
    for hp_name in args['hyper_params']:
        param_values.append(args[hp_name])

    combinations = list(itertools.product(*param_values))
    n_combinations = len(combinations)

    if idx >= n_combinations:
        print("Out of bounds for Hyper param settings. Terminating...")
        exit()

    pids = [None] * n_parallel_threads
    f = [None] * n_parallel_threads
    last_process = False

    # Start all the parallel threads on SINGLE GPU assigned to this script by Slurm
    # Warning: If Multiple GPUs are assigned by slurm, they will be unused
    for i in range(idx, idx + n_parallel_threads):
        setting = combinations[i]

        # Unroll the 'algos' parameters into the settings dictionary
        setting = dict(zip(args['hyper_params'], setting))
        setting.update(dict(zip(format, setting['algos'])))
        del setting['algos']

        # Set Hyper-parameters
        name = ''
        for temp in format:
            name += ' ' + str(setting[temp])

        now = datetime.now()
        timestamp = name + str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second)  # +':'+str(now.microsecond)
        args['timestamp'] = [timestamp]

        # Create Args Directory to save arguments
        if not path.exists(args_path):
            create_directory_tree(str.split(args_path, sep='/')[:-1])
        np.save(path.join(args_path, name), args)

        # Create Log Directory for stdout Dumps
        if not path.exists(stdout_dump_path):
            create_directory_tree(str.split(stdout_dump_path, sep='/')[:-1])

        # Create command
        command = "python ../../src/__main__.py "  #TODO

        folder_suffix = ''
        for name, value in setting.items():
            command += "--" + name + " " + str(value) + " "
            if name != 'dataset':
                folder_suffix += "_" + str(value)
        command += "--" + "folder_suffix " + folder_suffix + '__' + str(i + 1) + '/' + str(n_combinations)
        print(i + 1, '/', n_combinations, command)

        name = path.join(stdout_dump_path, folder_suffix)
        with open(name, 'w') as f[i]:
            pids[i] = subprocess.Popen(command.split(), stdout=f[i])
        time.sleep(3)

        if i == n_combinations - 1:
            break

    # Wait for all the parallel processes before exiting
    start = datetime.now()
    print('########## Waiting #############')
    for t in range(i, idx-1, -1):
        pids[i - t].wait()
    end = datetime.now()
    print('########## Waiting Over######### Took', diff(end, start), 'for', n_parallel_threads, 'threads')

else:
    # name = machine + 'simple_x_-_0_1_1'
    # timestamp = name + '9|8|5:56:26'  # '05|12|03:41'  # Month | Day | hours | minutes (24 hour clock)

    # Set Hyper-parameters
    for algo in args['algos']:
        name = args_path
        for temp in format:
            name += ' ' + str(algo[temp])

        try:
            args = np.load(name+'.npy').item()
            write_results(args)
            print("Done tabulation")
        except:
            print('model not found', name)


