#!/usr/bin/env python3
import fire
import pathlib
from pathlib import Path

import nltk
nltk.download('punkt')

from build_coha_corpus import build_train_test, build_data_sets

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_folders', type=str,
    #                     help='Path to COHA folders containing articles for each temporal slice separated by ";".',
    #                     default='data/coha/COHA_1960;data/coha/COHA_1990')
    # parser.add_argument('--output_files', type=str, help='Path to output files containing text for each'
    #                                                        'temporal slice separated by ";". Should correspond'
    #                                                        'to the number and order of input folders.',
    #                     default='data/coha/coha_1960.txt;data/coha/coha_1990.txt')
    # parser.add_argument('--lm_output_train', type=str,
    #                     help='Path to output .txt file used for language model training',
    #                     default='data/coha/train.txt')
    # parser.add_argument('--lm_output_test', type=str,
    #                     help='Path to output .txt file used for language model validation',
    #                     default='data/coha/test.txt')
    # args = parser.parse_args()

    # input_folders = args.input_folders.split(';')
    # output_files = args.output_files.split(';')

    # build_train_test(input_folders, args.lm_output_train, args.lm_output_test)
    # build_data_sets(input_folders, output_files)

def build_coha_corpus(dirpaths_for_input_slices:list):

    data_root_dir = Path('./data/')
    output_dir = data_root_dir / Path('outputs')

    input_folders = [ Path(str(dirpath)) for dirpath in dirpaths_for_input_slices ]
    output_files = [ output_dir / Path(str(Path(dirpath).stem) + '.txt') for dirpath in dirpaths_for_input_slices ]

    lm_output_train = output_dir / Path('train.txt')
    lm_output_test = output_dir / Path('test.txt')


    for e in [dirpaths_for_input_slices, input_folders, output_files, lm_output_train, lm_output_test]:
        print(e)
    # build_train_test(input_folders, lm_output_train, lm_output_test)
    # build_data_sets(input_folders, output_files)

def do_build_coha_corpus(): pass

if __name__ == '__main__':
    fire.Fire(build_coha_corpus)
