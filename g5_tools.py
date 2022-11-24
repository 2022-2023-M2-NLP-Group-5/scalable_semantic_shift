#!/usr/bin/env python3

from os import PathLike
from pathlib import Path
import pathlib

import fire
import sarge

from build_coha_corpus import build_train_test, build_data_sets

DEFAULT_DATA_ROOT_DIR='data/'

def _download_file_maybe(url:str, out_file_path=None):
    if out_file_path is None:
        out_file_path = Path(DEFAULT_DATA_ROOT_DIR) / Path(Path(url).name)

    if out_file_path.is_file():
        print('Not downloading because there is already a file at: {} ({})'.format(out_file_path, out_file_path.absolute()))
    else:
        # https://stackoverflow.com/questions/1078524/how-to-specify-the-download-location-with-wget
        command = sarge.shell_format('wget -O {} {} ', out_file_path, url)
        sarge.run(command)

def setup_punkt():
    '''Punkt tokenizer is used by the COHA preprocessing code (build_corpus...)'''
    import nltk
    nltk.download('punkt')

def _freeze_hashdeep_rel(data_dirpath=DEFAULT_DATA_ROOT_DIR, hashdeep_outfile_name='data.hashdeep.rel.txt'):
    command = ' hashdeep -revv -l "{}" > "{}" '
    command = sarge.shell_format(command, data_dirpath, hashdeep_outfile_name)
    print('Running command:', repr(command))
    sarge.run(command)
    # return p.returncode

def _freeze_hashdeep_bare(data_dirpath=DEFAULT_DATA_ROOT_DIR, hashdeep_outfile_name='data.hashdeep.bare.txt'):
    command = ' hashdeep -revv -b "{}" > "{}" '
    command = sarge.shell_format(command, data_dirpath, hashdeep_outfile_name)
    print('Running command:', repr(command))
    sarge.run(command)

def freeze_hashdeep(data_dirpath=DEFAULT_DATA_ROOT_DIR):
    _freeze_hashdeep_rel(data_dirpath)
    _freeze_hashdeep_bare(data_dirpath)

class coha(object):

    @staticmethod
    def download_sample():
        '''example CLI invocation: python g5_tools.py coha download_sample '''
        url = 'https://www.corpusdata.org/coha/samples/text.zip'
        out_file_path = Path(Path(url).name)
        if out_file_path.is_file():
            print('Not downloading because there is already a file at: {} ({})'.format(out_file_path, out_file_path.absolute()))
        else:
            command = sarge.shell_format('wget {}', url)
            sarge.run(command)

    @staticmethod
    def build_corpus(*dirpaths_for_input_slices, data_root_dir=DEFAULT_DATA_ROOT_DIR):
        '''Wrapper alternative to simplify the argparse version in the original build_coha_corpus.py.

        Invoke like so:
        python g5_tools.py coha build_corpus "data/coha/coha_1883/"  "data/coha/coha_1908/"
        '''

        if __name__ == '__main__':
            setup_punkt()

        data_root_dir = Path(data_root_dir)
        output_dir = data_root_dir / Path('outputs')

        input_folders = [ Path(dirpath) for dirpath in dirpaths_for_input_slices ]

        corpus_slice_labels = [ str(Path(dirpath).name) for dirpath in dirpaths_for_input_slices ]

        paths_for_lm_output_train = [ output_dir / Path(l) / Path('train.txt') for l in corpus_slice_labels ]
        paths_for_lm_output_test =  [ output_dir / Path(l) / Path('test.txt') for l in corpus_slice_labels ]

        print('\nWorking with the following paths...')
        for e in [dirpaths_for_input_slices, input_folders, paths_for_lm_output_train, paths_for_lm_output_test]:
            print(e)

        print('\nOk, making changes to filesystem...')
        output_dir.mkdir(exist_ok=True) # create outputs dir if it does not exist yet
        for infolder, lm_output_train, lm_output_test in zip(input_folders, paths_for_lm_output_train, paths_for_lm_output_test):
            build_train_test([infolder], lm_output_train, lm_output_test)
        # build_data_sets(input_folders, output_files) # this function outputs json files, which we have no use for


class bert(object):

    @staticmethod
    def download_model(url = 'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip'):
        '''NOTE: don't use this for now'''
        _download_file_maybe(url)

    @staticmethod
    def train(train = '~/projlogiciel/scalable_semantic_shift/data/outputs/coha.coha_1883.train.txt',
              out = '~/projlogiciel/scalable_semantic_shift/data/outputs/bert_training/',
              test = '~/projlogiciel/scalable_semantic_shift/data/outputs/coha.coha_1883.test.txt',
    ):
        """

        CLI invocation as explained in the main README:
        python fine-tune_BERT.py --train_data_file pathToLMTrainSet --output_dir pathToOutputModelDir --eval_data_file pathToLMTestSet --model_name_or_path modelForSpecificLanguage --mlm --do_train --do_eval --evaluate_during_training
        """

        """╰─λ ls -lah ~/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/                                                  
total 684M
drwxr-xr-x 2 user user 4.0K Nov 24 22:24 ./
drwxr-xr-x 6 user user 4.0K Nov 24 21:54 ../
-rw-r--r-- 1 user user  521 Nov 24  2018 bert_config.json
-rw-r--r-- 1 user user 682M Nov 24  2018 bert_model.ckpt.data-00000-of-00001
-rw-r--r-- 1 user user 8.5K Nov 24  2018 bert_model.ckpt.index
-rw-r--r-- 1 user user 888K Nov 24  2018 bert_model.ckpt.meta
lrwxrwxrwx 1 user user   97 Nov 24 22:18 config.json -> /home/user/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_config.json
lrwxrwxrwx 1 user user  102 Nov 24 22:24 model.ckpt.index -> /home/user/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_model.ckpt.index
-rw-r--r-- 1 user user 973K Nov 24  2018 vocab.txt
"""

        command = '''python fine-tune_BERT.py \
        --train_data_file {pathToLMTrainSet} \
        --output_dir {pathToOutputModelDir} \
        --eval_data_file {pathToLMTestSet} \
        --model_name_or_path {modelForSpecificLanguage} \
        --mlm --do_train --do_eval --evaluate_during_training

        '''
        # --config_name {pathToCfg}

        cmd = sarge.shell_format(command,
                                 pathToLMTrainSet = train,
                                 pathToOutputModelDir = out,
                                 pathToLMTestSet = test,

                                 # this works, it will download it from internet
                                 modelForSpecificLanguage = 'bert-base-multilingual-cased'

                                 # Here are other various notes from when i tried to make it work via loading a local, previously-downloaded model...

                                 # NOTE: '~' did not seem to work, absolute path needed
                                 # modelForSpecificLanguage = '/home/user/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_model.ckpt.data-00000-of-00001',

                                 # pathToCfg = '/home/user/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_config.json',
        )


        # Also tried (with partial success):
        # ln -s ~/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_config.json ~/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/config.json

        sarge.run(cmd)



if __name__ == '__main__':
    fire.Fire()

