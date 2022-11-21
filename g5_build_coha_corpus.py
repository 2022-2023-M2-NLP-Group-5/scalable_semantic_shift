#!/usr/bin/env python3

import fire
from pathlib import Path
import nltk

from build_coha_corpus import build_train_test, build_data_sets

def build_coha_corpus(dirpaths_for_input_slices:list, data_root_dir='./data/'):
    '''Wrapper alternative to simplify the argparse version in the original build_coha_corpus.py.

    Invoke like so: python3 g5_build_coha_corpus.py '["data/coha/coha_1883","data/coha/coha_1908"]'

    (mind the extra pair of quotes, which may be necessary according to your version of python, fire, shell, etc.)
    '''

    if __name__ == '__main__':
        nltk.download('punkt')

    data_root_dir = Path(data_root_dir)
    output_dir = data_root_dir / Path('outputs') # TBD: make this work if outputs dir does not exist yet

    input_folders = [ Path(dirpath) for dirpath in dirpaths_for_input_slices ]
    output_files = [ output_dir / Path(str(Path(dirpath).stem) + '.txt') for dirpath in dirpaths_for_input_slices ]

    lm_output_train = output_dir / Path('coha.train.txt')
    lm_output_test = output_dir / Path('coha.test.txt')

    print('\nWorking with the following paths...')
    for e in [dirpaths_for_input_slices, input_folders, output_files, lm_output_train, lm_output_test]:
        print(e)

    print('\nOk, making changes to filesystem...')
    build_train_test(input_folders, lm_output_train, lm_output_test)
    build_data_sets(input_folders, output_files)

if __name__ == '__main__':
    fire.Fire(build_coha_corpus)
