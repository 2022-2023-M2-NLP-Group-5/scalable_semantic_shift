#!/usr/bin/env python3
'''
Prep:

#oarsub -l gpu=1 -I -q production # nancy
oarsub -l gpu=2 -I -t exotic # grenoble, lyon

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh

git clone https://github.com/2022-2023-M2-NLP-Group-5/scalable_semantic_shift/
cd scalable_semantic_shift/
git checkout devel

time mamba env create -f environment.yml  # 4 mins

conda activate ScaleSemShift
time python g5_tools.py bert prep_coha
time python g5_tools.py bert train

'''

import sys
from pathlib import Path
import shutil
import glob

import fire
import sarge

# import pathlib
# from six import class_types, reraise

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level='TRACE',
               backtrace=True, diagnose=True)
except:
    import logging as logger
    logger.warning('Did not find `loguru` module, defaulting to `logging`')

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
    command = ' hashdeep -c sha256 -revv -l "{}" > "{}" '
    command = sarge.shell_format(command, data_dirpath, hashdeep_outfile_name)
    print(f'Running {command=}')
    sarge.run(command)
    # return p.returncode

def _freeze_hashdeep_bare(data_dirpath=DEFAULT_DATA_ROOT_DIR, hashdeep_outfile_name='data.hashdeep.bare.txt'):
    command = ' hashdeep -c sha256 -revv -b "{}" > "{}" '
    command = sarge.shell_format(command, data_dirpath, hashdeep_outfile_name)
    print(f'Running {command=}')
    sarge.run(command)

def freeze_hashdeep(data_dirpath=DEFAULT_DATA_ROOT_DIR):
    _freeze_hashdeep_rel(data_dirpath)
    _freeze_hashdeep_bare(data_dirpath)

class coha(object):

    ZIP_EXTRACTION_DEST_DIR = Path(DEFAULT_DATA_ROOT_DIR) / 'text.zip.coha.d/'
    SLICE_EXTRACTION_DEST_DIR_ROOT = Path(DEFAULT_DATA_ROOT_DIR) / 'coha'

    @staticmethod
    def download_sample():
        '''example CLI invocation: python g5_tools.py coha download_sample '''
        url = 'https://www.corpusdata.org/coha/samples/text.zip'
        _download_file_maybe(url)

    @classmethod
    def _unzip_dataset(cls,
                       exdir=None,
                       zipfile_path = Path(DEFAULT_DATA_ROOT_DIR) / Path('text.zip')
                       ):
        if exdir is None: exdir = cls.ZIP_EXTRACTION_DEST_DIR
        try:
            exdir.mkdir(exist_ok=False, parents=True)
        except:
            logger.warning(f'Aborting because {exdir=} ...seems to already exist')
            sys.exit(1)
        command = sarge.shell_format('unzip -q -d {} {} ', exdir, zipfile_path)
        sarge.run(command)

    @classmethod
    def clear(cls):
        '''clear_all_except_zip. and also does not clear the output of build_corpus'''
        for e in [cls.ZIP_EXTRACTION_DEST_DIR,
                  cls.SLICE_EXTRACTION_DEST_DIR_ROOT]:
            logger.info(f'Removing: {e}')
            shutil.rmtree(e, ignore_errors=True)

    @classmethod
    def extract_one_slice(cls,
                          slice_pattern:str='1910',
                          ):
        """Example equivalent operation: mv data/text.zip.coha.d/*1910*.txt data/coha/1910/"""
        # dir where full dataset was unzipped to
        exdir = cls.ZIP_EXTRACTION_DEST_DIR

        slice_pattern = str(slice_pattern)
        logger.trace(f'{slice_pattern=}')

        # dir where we will store our extracted slices
        slice_out_dir = cls.SLICE_EXTRACTION_DEST_DIR_ROOT / (slice_pattern + '/')
        slice_out_dir.mkdir(parents=True)

        found_matches = False
        fpaths = exdir.glob(('*_' + slice_pattern + '_*.txt'))
        for fpath in fpaths:
            found_matches = True # if we found at least one match for the pattern
            # NOTE: mv is faster than cp, but harms reproducibility/idempotency
            command = sarge.shell_format('mv {} {}', fpath, slice_out_dir)
            logger.info(f'{command=}')
            sarge.run(command)

        if not found_matches:
            logger.warning('No matching files')

    @classmethod
    def _extract_several_slices(cls, *slices_patterns):
        logger.trace(f'{slices_patterns=}')
        for patt in slices_patterns:
            cls.extract_one_slice(slice_pattern=patt)

    @classmethod
    def extract(cls, *slices_patterns):
        logger.trace(f'{slices_patterns=}')
        cls._unzip_dataset()
        cls._extract_several_slices(*slices_patterns)

    @staticmethod
    def build_corpus(*dirpaths_for_input_slices, data_root_dir=DEFAULT_DATA_ROOT_DIR):
        '''Wrapper alternative to simplify the argparse version in the original build_coha_corpus.py.

        Invoke like so:
        python g5_tools.py coha build_corpus "data/coha/coha_1883/"  "data/coha/coha_1908/"
        '''

        if __name__ == '__main__':
            setup_punkt()

        # this import needs punkt already downloaded in order to succeed
        from build_coha_corpus import build_train_test, build_data_sets

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
            lm_output_train.parent.mkdir(exist_ok=True, parents=True)
            lm_output_test.parent.mkdir(exist_ok=True)
            build_train_test([infolder], lm_output_train, lm_output_test)
        # build_data_sets(input_folders, output_files) # this function outputs json files, which we have no use for


class bert(object):

    @staticmethod
    def download_model(url = 'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip'):
        '''NOTE: don't use this for now'''
        _download_file_maybe(url)

    @staticmethod
    def prep_coha():
        coha.download_sample()
        coha.clear()
        coha.extract('1910', '1950')
        coha.build_corpus("data/coha/1910/","data/coha/1950/")
        coha.clear()

    @staticmethod
    def train(
            train = 'data/outputs/1910/train.txt',
            out = 'data/outputs/bert_training/',
            test = 'data/outputs/1910/test.txt',
            batchSize = 6,

            # train = '~/projlogiciel/scalable_semantic_shift/data/outputs/coha.coha_1883.train.txt',
            # out = '~/projlogiciel/scalable_semantic_shift/data/outputs/bert_training/',
            # test = '~/projlogiciel/scalable_semantic_shift/data/outputs/coha.coha_1883.test.txt',
    ):
        """
        Example invocation:

        oarsub -l gpu=1 -I -q production  # and wait for it to connect
        bash
        conda activate ScaleSemShift
        cd ~/s9/software-project/repo/scalable_semantic_shift
        ./g5_tools.py bert train --train data/outputs/coha.1910.train.txt  --out data/outputs/bert_training  --test data/outputs/coha.1910.test.txt


        Original CLI invocation that this function wraps, as explained in the main README:

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
        --mlm --do_train --do_eval --evaluate_during_training \
        --per_gpu_train_batch_size {batchSize} \
        --per_gpu_eval_batch_size {batchSize}


        '''
        # --config_name {pathToCfg}

        cmd = sarge.shell_format(command,
                                 pathToLMTrainSet = train,
                                 pathToOutputModelDir = out,
                                 pathToLMTestSet = test,

                                 # this works, it will download it from internet
                                 modelForSpecificLanguage = 'bert-base-multilingual-cased',

                                 batchSize = batchSize,

                                 # Here are other various notes from when i tried to make it work via loading a local, previously-downloaded model...

                                 # NOTE: '~' did not seem to work, absolute path needed
                                 # modelForSpecificLanguage = '/home/user/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_model.ckpt.data-00000-of-00001',

                                 # pathToCfg = '/home/user/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_config.json',
        )


        # Also tried (with partial success):
        # ln -s ~/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_config.json ~/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/config.json

        sarge.run(cmd)

@logger.catch
def main():
    fire.Fire()

if __name__ == '__main__':
    main()
