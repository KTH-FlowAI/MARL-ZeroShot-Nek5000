#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from configs_meta import add_subparser as add_subparser_conf
from initial_meta import add_subparser as add_subparser_initial
from eval_meta    import add_subparser as add_subparser_eval
parser = argparse.ArgumentParser(
    prog="python -m meta_MARL", description="Meta Polices for evaluating DRL using NEK5000 for drag reduction"
)

subparsers = parser.add_subparsers()
add_subparser_initial(subparsers)
add_subparser_eval(subparsers)
add_subparser_conf(subparsers)
args = parser.parse_args()
args.cmd(**vars(args))
