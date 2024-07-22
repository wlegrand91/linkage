
import linkage

import argparse
import sys

def main_cli(argv=None):
    
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog='linkage',
                                     description='What the program does',
                                     epilog='Text at the bottom of help')
    
    args = parser.parse_args(argv)

    # run program on args here

    