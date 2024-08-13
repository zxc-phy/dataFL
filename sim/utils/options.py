"""
This file is used to parse the arguments from the command line.
"""
import argparse
import numpy as np
import random

class SingleOrList(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) == 1:
            # When there is only one value, it is parsed as a single floating point number
            setattr(namespace, self.dest, float(values[0]))
        else:
            # When there are multiple values, they are parsed as a list of floating point numbers
            setattr(namespace, self.dest, [float(v) for v in values])

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', default='60', type=int, help='Number of SNs to collect data from')
    parser.add_argument('-N', default='5', type=int, help='Number of UAV or vehicles')
    parser.add_argument('-E', default='100', type=int, help='Number of episodes')
    # parser.add_argument('-m', default='vgg11', type=str, help='Model')
    parser.add_argument('-m', default='cnncifar', type=str, help='Model')
    parser.add_argument('-d', default='cifar10', type=str, help='Dataset')
    # parser.add_argument('-s', default=2, type=int, help='Index of split layer')
    parser.add_argument('-R', default=2, type=int, help='Number of total training rounds')
    parser.add_argument('-K', default=5, type=int, help='Number of local steps')
    parser.add_argument('-Zmax', default=4, type=int, help='max increasing episode of data')
    # parser.add_argument('-M', default=100, type=int, help='Number of total clients')
    # parser.add_argument('-P', default=100, type=int, help='Number of clients participate')
    parser.add_argument('--partition', default='dir', type=str, choices=['dir', 'iid', 'exdir'], 
                        help='Data partition')
    parser.add_argument('--alpha', default=30.0, type=float, nargs='*', action=SingleOrList,
                         help='The parameter `alpha` of dirichlet distribution')
    # parser.add_argument('--alpha', default=10, type=float, nargs='*',
    #                      help='The parameter `alpha` of dirichlet distribution')
    parser.add_argument('--optim', default='sgd', type=str, 
                        choices=['sgd', 'adam'], help='Optimizer')
    parser.add_argument('--lr', default=0.03, type=float, help='Client/Local learning rate')
    parser.add_argument('--lr-decay', default=1.0, type=float, help='Learning rate decay')
    parser.add_argument('--momentum', default=0.0, type=float, help='Momentum of client optimizer')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay of client optimizer')
    parser.add_argument('--global-lr', default=1.0, type=float, help='Server/Global learning rate')
    parser.add_argument('--batch-size', default=50, type=int, help='Mini-batch size')
    parser.add_argument('--seed', default=1234, type=int, help='Seed')
    parser.add_argument('--clip', default=0, type=int, help='Clip')
    parser.add_argument('--log', default='', type=str, help='Log, Log/Print')
    parser.add_argument('--eval-num', default=1000, type=int, help='Number of evaluations')
    parser.add_argument('--tail-eval-num', default=0, type=int, help='Evaluating the tail # rounds')
    parser.add_argument('--device', default=0, type=int, help='Device')
    parser.add_argument('--save-model', default=0, type=int, help='Whether to save model')
    parser.add_argument('--start-round', default=0, type=int, help='Start')
    parser.add_argument('--datatype', default=[1, 1, 0, 0, 1, 1, 1, 2, 0, 0, 1, 1, 0, 0, 2, 1, 2, 0, 2, 1, 2, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 2, 2, 2, 1, 1, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 2, 0, 2, 1, 0, 0, 2, 0, 2, 2, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 0, 0, 0, 2, 1, 0, 2, 2, 0, 2, 1, 0, 0, 2, 0, 2, 0, 1, 0, 0, 2, 1, 0, 2, 2, 1, 2, 0, 1, 2, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 0, 2, 0, 1, 0, 1, 2, 1, 0, 1, 0, 0, 2, 0, 2, 2, 1, 0, 1, 0, 0, 1, 2, 0, 0, 1, 1, 2, 0, 2, 2, 2, 1, 2, 1, 1, 2, 1], type=list,help='Data type of each SN')
    parser.add_argument('--noise', default=[0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.3, 0, 0, 0.5, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], type=list, help='Noise')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    pass
    


# [0, 0, 0.3, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.3, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.3, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0.3, 0, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.3, 0, 0]
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.3, 0, 0, 0.5, 0, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0, 0, 0.5, 0.5, 0.5, 0, 0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0]