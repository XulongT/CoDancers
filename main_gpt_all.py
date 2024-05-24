from mc_gpt_all import MCTall 
import argparse
import os
import yaml
from pprint import pprint
from easydict import EasyDict



def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of Music2Dance')
    parser.add_argument('--config', default='')
    parser.add_argument('--theta', type=int, default=2000)
    parser.add_argument('--k', type=int, default=1)
    # exclusive arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--eval', action='store_true')
    group.add_argument('--visgt', action='store_true')
    group.add_argument('--anl', action='store_true')
    group.add_argument('--sample', action='store_true')
    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in vars(args).items():
        config[k] = v

    config = EasyDict(config)
    config.structure_generate.theta = args.theta
    config.structure_generate.k = args.k

    if args.train:
        config.update({'need_not_test_data':True})
    elif args.eval:
        config.update({'need_not_train_data':True})
        config.update({'need_not_eval_data':True})

    agent = MCTall(config)
    print(config)

    if args.train:
        agent.train()
    elif args.eval:
        agent.eval()
    elif args.visgt:
        agent.visgt()
    elif args.anl:
        agent.analyze_code()
    elif args.sample:
        config.update({'need_not_train_data':True})
        config.update({'need_not_test_data':True})
        agent.sample()


if __name__ == '__main__':
    main()
