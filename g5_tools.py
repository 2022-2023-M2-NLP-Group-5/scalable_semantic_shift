#!/usr/bin/env python3
'''Collection of tools for our purposes.

Example prep for new node/cluster (if you're running on a cluster that is
already set up, then you just need some of these commands):

#oarsub -l gpu=1 -I -q production # nancy
oarsub -l gpu=2 -I -t exotic # grenoble, lyon

oarsub -I -p "cluster='gemini'" -l gpu=1,walltime=1:00  -t exotic  # lyon

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh

bash # enter a new bash shell with conda/mamba available

git clone https://github.com/2022-2023-M2-NLP-Group-5/scalable_semantic_shift/
cd scalable_semantic_shift/
git checkout devel

time mamba env create -f environment.yml  # 4 mins (mostly for scikit)

bash

conda activate ScaleSemShift
time python g5_tools.py bert prep_coha
time python g5_tools.py bert train

'''

import sys
from pathlib import Path
import shutil
import glob

from numpy import mean
import pandas as pd

import fire
import sarge

try:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, backtrace=True, diagnose=True, level='TRACE')
    logger.add("logs/g5.{time}.loguru.log", retention=50, level='TRACE')
except:
    import logging as logger
    logger.warning('Did not find `loguru` module, defaulting to `logging`')

DEFAULT_DATA_ROOT_DIR = Path('data/')

def _download_file_maybe(url:str, out_file_path=None):
    if out_file_path is None:
        out_file_path = Path(DEFAULT_DATA_ROOT_DIR) / Path(Path(url).name)

    if out_file_path.is_file():
        logger.warning('Not downloading because there is already a file at: {} ({})'.format(out_file_path, out_file_path.absolute()))
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
    logger.info(f'Running {command=}')
    sarge.run(command)
    # return p.returncode

def _freeze_hashdeep_bare(data_dirpath=DEFAULT_DATA_ROOT_DIR, hashdeep_outfile_name='data.hashdeep.bare.txt'):
    command = ' hashdeep -c sha256 -revv -b "{}" > "{}" '
    command = sarge.shell_format(command, data_dirpath, hashdeep_outfile_name)
    logger.info(f'Running {command=}')
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
        logger.info(f'{slice_pattern=}')

        # dir where we will store our extracted slices
        slice_out_dir = cls.SLICE_EXTRACTION_DEST_DIR_ROOT / (slice_pattern + '/')
        slice_out_dir.mkdir(parents=True)

        found_matches = False
        fpaths = exdir.glob(('*_' + slice_pattern + '_*.txt'))
        for fpath in fpaths:
            found_matches = True # if we found at least one match for the pattern
            # NOTE: mv is faster than cp, but harms reproducibility/idempotency
            command = sarge.shell_format('mv {} {}', fpath, slice_out_dir)
            logger.debug(f'{command=}')
            sarge.run(command)

        if not found_matches:
            logger.warning('No matching files')

    @classmethod
    def _extract_several_slices(cls, *slices_patterns):
        logger.debug(f'{slices_patterns=}')
        for patt in slices_patterns:
            cls.extract_one_slice(slice_pattern=patt)

    @classmethod
    def extract(cls, *slices_patterns):
        logger.debug(f'{slices_patterns=}')
        cls._unzip_dataset()
        cls._extract_several_slices(*slices_patterns)

    @staticmethod
    def build_corpus(*dirpaths_for_input_slices,
                     data_root_dir=DEFAULT_DATA_ROOT_DIR,
                     do_txt:bool=True,
                     do_json:bool=True):
        '''Wrapper alternative to simplify the argparse version in the original build_coha_corpus.py.
        '''
        logger.debug(f'{do_txt=}, {do_json=}')

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

        json_output_files = [ output_dir / Path(l) / Path('full_text.json.txt') for l in corpus_slice_labels ]

        logger.debug('Working with the following paths...')
        for el in [dirpaths_for_input_slices,
                   input_folders,
                   paths_for_lm_output_train,
                   paths_for_lm_output_test,
                   json_output_files]:
            logger.debug(el)

        logger.info('Ok, making changes to filesystem...')
        output_dir.mkdir(exist_ok=True) # create outputs dir if it does not exist yet

        if do_txt:
            logger.info('Running the loop for `build_train_test()`')
            for infolder, lm_output_train, lm_output_test in zip(input_folders, paths_for_lm_output_train, paths_for_lm_output_test):
                logger.info(f'{infolder=}')
                logger.debug(f'{lm_output_train=}, {lm_output_test=}')
                lm_output_train.parent.mkdir(exist_ok=True, parents=True)
                lm_output_test.parent.mkdir(exist_ok=True)
                build_train_test([infolder], lm_output_train, lm_output_test)

        if do_json:
            # This function outputs json files. For COHA, these files are what
            # get_embeddings_scalable.get_slice_embeddings fn expects.
            logger.info('Running `build_data_sets()`')
            build_data_sets(input_folders, json_output_files)

    @classmethod
    def prep_slices(cls, *slices):
        """This functions wraps the other ones in this class for easy usage. It will
perform all-in-one: download sample, clear some data workspace, extract the
slices, build the corpuses from the slices, cleanup afterwards.

        """
        logger.debug(f'{slices=}')
        slices = [str(l) for l in slices]
        logger.info(f'{slices=}')
        slice_dirs = [cls.SLICE_EXTRACTION_DEST_DIR_ROOT / l for l in slices]
        logger.debug(f'{slice_dirs=}')

        cls.download_sample()
        cls.clear()
        cls.extract(*slices)
        cls.build_corpus(*slice_dirs)
        cls.clear()

# might as well hardcode this since the list is so short
SEMEVAL_WORDLIST = [ 'attack', 'bag', 'ball', 'bit', 'chairman',
                     'circle', 'contemplation', 'donkey', 'edge', 'face', 'fiction', 'gas', 'graft',
                     'head', 'land', 'lane', 'lass', 'multitude', 'ounce', 'part', 'pin', 'plane',
                     'player', 'prop', 'quilt', 'rag', 'record', 'relationship', 'risk', 'savage',
                     'stab', 'stroke', 'thump', 'tip', 'tree', 'twist', 'word', ]

class bert(object):

    @staticmethod
    def download_model(url = 'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip'):
        '''NOTE: don't use this for now. Instead code is using the model that hugging face libraries will download automagically'''
        _download_file_maybe(url)

    @staticmethod
    def prep_coha(*slices):
        if len(slices) == 0:
             slices = list(range(1910,2010)) # Before 1900, the COHA sample corpus coverage is spotty

        coha.prep_slices(*slices)

        # coha.download_sample()
        # coha.clear()
        # coha.extract('1910', '1950')
        # coha.build_corpus("data/coha/1910/","data/coha/1950/")
        # coha.clear()

    @staticmethod
    def train(
            train = 'data/outputs/1910/train.txt',
            out = 'data/outputs/bert_training/',
            test = 'data/outputs/1910/test.txt',
            batchSize = 7,
            epochs = 5.0, # 5.0 is the default they use
            # **kwargs, # TBD: figure out how to implement generic kwargs passthru
    ):
        """Wraps `fine-tune_BERT.py` in order to run a test training.

        Original CLI invocation that this function wraps, as explained in the main README:

        python fine-tune_BERT.py --train_data_file pathToLMTrainSet --output_dir pathToOutputModelDir --eval_data_file pathToLMTestSet --model_name_or_path modelForSpecificLanguage --mlm --do_train --do_eval --evaluate_during_training
        """

        command = '''python fine-tune_BERT.py \
        --train_data_file {pathToLMTrainSet} \
        --output_dir {pathToOutputModelDir} \
        --eval_data_file {pathToLMTestSet} \
        --model_name_or_path {modelForSpecificLanguage} \
        --mlm --do_train --do_eval --evaluate_during_training \
        \
        --per_gpu_train_batch_size {batchSize} \
        --per_gpu_eval_batch_size {batchSize} \
        --num_train_epochs {epochs}
        '''
        # --config_name {pathToCfg}

        cmd = sarge.shell_format(command,
                                 pathToLMTrainSet = train,
                                 pathToOutputModelDir = out,
                                 pathToLMTestSet = test,

                                 # this works, it will download it from internet
                                 modelForSpecificLanguage = 'bert-base-multilingual-cased',

                                 batchSize = batchSize,
                                 epochs = epochs,

                                 # Here are other various notes from when i tried to make it work via loading a local, previously-downloaded model...

                                 # NOTE: '~' did not seem to work, absolute path needed
                                 # modelForSpecificLanguage = '/home/user/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_model.ckpt.data-00000-of-00001',

                                 # pathToCfg = '/home/user/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_config.json',
        )

        # Also tried (with partial success):
        # ln -s ~/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_config.json ~/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/config.json

        logger.info(f'{cmd=}')
        sarge.run(cmd)

    @staticmethod
    def _get_gulordava_dict():
        '''between get_embeddings_scalable.py and get_embeddings_scalable_semeval.py, they have a lot in common but seem to have two functions which are renamed/equivalent:

        get_embeddings_scalable.py has get_shifts(), get_slice_embeddings()

        get_embeddings_scalable_semeval.py has get_targets(), get_time_embeddings()

        get_targets() is designed to take the targets file from semeval zip, and has logic to strip the POS suffixes (_nn etc) from the list of words.

        whereas get_shifts() expects the Gulordava csv file, and gets its list of words from there.
        '''
        # from get_embeddings_scalable import get_shifts # Gulordava csv
        # from get_embeddings_scalable_semeval import get_targets # SemEval targets file

        # https://marcobaroni.org/PublicData/gulordava_GEMS_evaluation_dataset.csv
        GULORDAVA_FILEPATH = (DEFAULT_DATA_ROOT_DIR / 'gulordava_GEMS_evaluation_dataset.csv').resolve()
        logger.info(f'{GULORDAVA_FILEPATH=}')

        # shifts_dict = get_shifts(GULORDAVA_FILEPATH)
        '''get_embeddings_scalable.get_shifts() is not compatible with this upstream version of the file; it seems(?) to expect format like "3 users" (mean_score:int word:str)'''

        shifts_dict = {}
        df_shifts = pd.read_csv(GULORDAVA_FILEPATH, sep=',', encoding='utf8')
        for idx, row in df_shifts.iterrows():
            ratings = [ row[l] for l in ['p1','p2','p3','p4','p5'] ]
            shifts_dict[row[0]] = mean(ratings)
        return shifts_dict

    @classmethod
    def _get_wordlist_for_extract_query(cls) -> list:
        gulordava_wordlist = list(cls._get_gulordava_dict().keys())

        wordlist = sorted(list(set(SEMEVAL_WORDLIST + gulordava_wordlist)))
        # wordlist = wordlist[:3] # for testing

        logger.info(f'{len(wordlist)=}, {wordlist=}')
        return wordlist

    @staticmethod
    def _make_mockup_dict_from_wordlist(wordlist:list) -> dict:
        '''Should output something in this format: shifts_dict = {'cat': '42', 'dog': '42', 'bird': '42', 'house': '42', }

        Based on input like: wordlist = ['cat', 'dog', 'bird', 'house']

        '''

        '''This shifts_dict setup is necessary because the original function expects a file containing tokens and frequencies(?).

        Default path listed in `get_embeddings_scalable.py` is like so:

        parser.add_argument("--target_path", default='data/coha/Gulordava_word_meaning_change_evaluation_dataset.csv', type=str, help="Path to target file")

        This file does not seem to be readily available online, and in any case we don't necessarily want to restrict the set of words we look at in this same way.

        The functions in `get_embeddings_scalable.py` don't seem to actually do anything with the frequency info, so I have just included dummy values to fit the datastructure format.

        '''
        mockup_dict = {}
        for w in wordlist:
            mockup_dict[w] = 42
        return mockup_dict

    @classmethod
    def _get_mockup_dict_for_extract_query(cls):
        mockup_dict = cls._make_mockup_dict_from_wordlist( cls._get_wordlist_for_extract_query() )
        logger.debug(f'{mockup_dict=}')
        return mockup_dict

    @classmethod
    def extract(cls,
                pathToFineTunedModel:str='data/RESULTS_train_bert_coha/1910/pytorch_model.bin',
                dataset:str='data/outputs/1910/full_text.json.txt',
                gpu=True):
        """
        Wraps get_embeddings_scalable.py.
        Extract embeddings from the preprocessed corpus in .txt for corpus.

        The original repo's functions are designed for a single monomodel, in which multiple slices/corpuses are present and can be extracted.

        From original:

python get_embeddings_scalable.py --corpus_paths pathToPreprocessedCorpusSlicesSeparatedBy';' --corpus_slices nameOfCorpusSlicesSeparatedBy';' --target_path pathToTargetFile --task chooseBetween'coha','durel','aylien' --path_to_fine_tuned_model pathToFineTunedModel --embeddings_path pathToOutputEmbeddingFile

This creates a pickled file containing all contextual embeddings for all target words.
        """
        # pathToPreprocessedCorpusSlices = ''
        # nameOfCorpusSlices = ''

        # cmd = f'''python get_embeddings_scalable.py \
        # --corpus_paths pathToPreprocessedCorpusSlices \
        # --corpus_slices nameOfCorpusSlices \
        # --target_path pathToTargetFile \
        # --task 'coha' \
        # --path_to_fine_tuned_model {pathToFineTunedModel} \
        # --embeddings_path pathToOutputEmbeddingFile \

        # '''

        logger.info('Working with the following input parameters...')
        logger.info(f'{pathToFineTunedModel=}')
        logger.info(f'{dataset=}')
        logger.info(f'{gpu=}')

        import torch
        from transformers import BertTokenizer, BertModel
        from get_embeddings_scalable import get_slice_embeddings

        batch_size = 16
        max_length = 256

        # slices = args.corpus_slices.split(';')
        # slices = ['1910', '1950']

        # lang = 'English'
        task = 'coha'

        # task = args.task

        # tasks = ['coha', 'aylien', 'durel']
        # if task not in tasks:
        #     print("Task not valid, valid choices are: ", ", ".join(tasks))
        #     sys.exit()

        # datasets = args.corpus_paths.split(';')

        # datasets = ['data/outputs/1910/full_text.json.txt',
        #             'data/outputs/1950/full_text.json.txt']

        datasets = [dataset]
        slices = [str(Path(dataset).parent.stem)]

        logger.info(f'{datasets=} ; {slices=}')

        # embeddings_path: is path to output the embeddings file
        embeddings_path = DEFAULT_DATA_ROOT_DIR / 'embeddings' / f'{slices[0]}.pickle'
        embeddings_path.parent.mkdir(exist_ok=True)

        embeddings_path = str(embeddings_path.resolve())

        logger.info(f'We will output embeddings to file: {embeddings_path=}')


        '''
        NOTE: These dataset paths correspond to the files output by build_coha_corpus.build_data_sets (for coha, these are json format -- unclear why).

        Cf. this section from build_coha_corpus.py:

        parser.add_argument('--output_files', type=str, help='Path to output files containing text for each'
                                                               'temporal slice separated by ";". Should correspond'
                                                               'to the number and order of input folders.',
                            default='data/coha/coha_1960.txt;data/coha/coha_1990.txt')

        build_train_test(input_folders, args.lm_output_train, args.lm_output_test)
        build_data_sets(input_folders, output_files)

        Cf. this section from get_embeddings_scalable.py:

        parser.add_argument("--corpus_paths",
                            default='data/coha/coha_1960.txt;data/coha/coha_1990.txt',
                            type=str,
                            help="Paths to all corpus time slices separated by ';'.")
        '''

        # if len(args.path_to_fine_tuned_model) > 0:
        #     fine_tuned = True
        # else:
        #     fine_tuned = False

        fine_tuned = True

        # datasets = args.corpus_paths.split(';')

        # if len(args.path_to_fine_tuned_model) > 0:
        #     state_dict =  torch.load(args.path_to_fine_tuned_model)

        if task == 'coha':
            lang = 'English'
            # shifts_dict = get_shifts(args.target_path)
            shifts_dict = cls._get_mockup_dict_for_extract_query()

        # elif task == 'aylien':
        #     lang = 'English'
        #     shifts_dict = get_shifts(args.target_path)
        # elif task == 'durel':
        #     lang = 'German'
        #     shifts_dict = get_durel_shifts(args.target_path)


        # if lang == 'English':
        #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        #     if fine_tuned:
        #         state_dict = torch.load(args.path_to_fine_tuned_model)
        #         model = BertModel.from_pretrained('bert-base-uncased', state_dict=state_dict, output_hidden_states=True)
        #     else:
        #         model = BertModel.from_pretrained('bert_base-uncased', output_hidden_states=True)

        # https://stackoverflow.com/questions/65882750/please-use-torch-load-with-map-location-torch-devicecpu
        #torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if gpu:
            pass
        else:
            torch_device = torch.device("cpu")
            logger.info(f'{torch_device=}')

        modelForSpecificLanguage = 'bert-base-multilingual-cased'
        logger.info(f'{modelForSpecificLanguage=}')

        tokenizer = BertTokenizer.from_pretrained(modelForSpecificLanguage, do_lower_case=False)
        if gpu:
            state_dict =  torch.load(pathToFineTunedModel)
        else:
            state_dict =  torch.load(pathToFineTunedModel, map_location=torch_device)
        model = BertModel.from_pretrained(modelForSpecificLanguage, state_dict=state_dict, output_hidden_states=True)

        # elif lang == 'German':
        #     tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        #     if fine_tuned:
        #         state_dict = torch.load(args.path_to_fine_tuned_model)
        #         model = BertModel.from_pretrained('bert-base-german-cased', state_dict=state_dict, output_hidden_states=True)
        #     else:
        #         model = BertModel.from_pretrained('bert-base-german-cased', output_hidden_states=True)

        if gpu:
            model.cuda()
        model.eval()

        logger.debug(f'{embeddings_path=}, {datasets=}, {tokenizer=}, (`model` too verbose to log here), {batch_size=}, {max_length=}, {lang=}, {shifts_dict=}, {task=}, {slices=}, {gpu=}')
        # logger.debug(f'{model=}')

        get_slice_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length, lang, shifts_dict, task, slices, gpu=gpu)


    @staticmethod
    def measure():
        '''
        Wraps measure_semantic_shift.py.

        Method WD or JSD: "Wasserstein distance or Jensen-Shannon divergence".
        '''
        # TBD implement this.
        # currently this step is instead implemented elsewhere, in measure_semantic_shift_merged.py
        pass


class eval(object):

    @staticmethod
    def _filter_results_file_with_wordlist(results_filepath:str, filter_list:list=SEMEVAL_WORDLIST):
        results_filepath = str(Path(results_filepath).resolve())
        logger.info(f"Reading results from {results_filepath=}")
        df = pd.read_table(results_filepath, sep=';', encoding='utf8')

        df = df[ df['word'].isin(filter_list) ]
        df = df.sort_values('word')

        logger.debug(f'Filtered dataframe...:\n{df.head(999)}')
        return df

    @classmethod
    def filter_results(cls,
                       results_filepath:str,
                       filter_type:str='semeval',
                       out_filepath=''):
        if filter_type == 'semeval':
            filter_list = SEMEVAL_WORDLIST
        elif filter_type == 'gulordava':
            filter_list = gulordava_wordlist = list(bert._get_gulordava_dict().keys())
        else:
            raise NotImplementedError(f'{filter_type=}')

        results_filepath = str(Path(results_filepath).resolve())
        df = cls._filter_results_file_with_wordlist(results_filepath, filter_list=filter_list)

        if out_filepath == '':
            outfile_name = Path(results_filepath).stem + f'.filtered_for_{filter_type}' + f'.csv'
            logger.debug(f'{outfile_name=}')
            out_filepath = Path(results_filepath).parent / outfile_name

        out_filepath = Path(out_filepath).resolve()
        logger.info(f"Saving filtered results into {out_filepath=}")
        df.to_csv(out_filepath, sep=';', encoding='utf-8', index=False)

@logger.catch
def main():
    fire.Fire()

if __name__ == '__main__':
    main()
