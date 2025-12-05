#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from configs import add_subparser as add_subparser_conf
from run import add_subparser as add_subparser_run
from evaluate import add_subparser as add_subparser_evaluate
from initial import add_subparser as add_subparser_initial
from transfer_learning import add_subparser as add_subparser_transfer_learning

def main():
    """Main entry point for the nek_MARL command-line interface."""
    parser = argparse.ArgumentParser(
        prog="nek-marl", description="DRL using NEK5000 for drag reduction"
    )

    subparsers = parser.add_subparsers()
    add_subparser_initial(subparsers)
    add_subparser_run(subparsers)
    add_subparser_transfer_learning(subparsers)
    add_subparser_evaluate(subparsers)
    add_subparser_conf(subparsers)
    args = parser.parse_args()
    args.cmd(**vars(args))

if __name__ == "__main__":
    main()
