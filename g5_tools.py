#!/usr/bin/env python3
"""Collection of tools for our purposes.

Example prep for new node/cluster (if you're running on a cluster that is
already set up, then you just need some of these commands):

oarsub -I -p "cluster='gemini'" -l gpu=1,walltime=1:00  -t exotic  # lyon

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh

bash # enter a new bash shell with conda/mamba available

git clone https://github.com/2022-2023-M2-NLP-Group-5/scalable_semantic_shift/
cd scalable_semantic_shift/
git checkout experimental # or git checkout devel # depending on context

time mamba env create -f environment.yml  # 4 mins (mostly for scikit)

bash

conda activate ScaleSemShift
"""

import sys
import os
from pathlib import Path
import shutil
import hashlib
import contextlib  # with contextlib.redirect_stdout(stream_stdout), contextlib.redirect_stderr(stream_stderr)

import time

from numpy import mean
import pandas as pd

import fire
import sarge
import pendulum

try:
    from loguru import logger

    logger.remove()
    # https://loguru.readthedocs.io/en/stable/api/logger.html?#loguru._logger.Logger.add
    logger_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>elapsed {elapsed}</level> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    # fmt: off
    logger.add( sys.stderr, backtrace=True, diagnose=True, level="TRACE", format=logger_fmt )
    logger.add( "logs/{time:YYYY-MM-DD}/g5.{time}.loguru.log", level="TRACE", retention="6 months", format=logger_fmt, colorize=True, )
    logger.add( "logs/{time:YYYY-MM-DD}/g5.{time}.loguru.log.txt",  level="TRACE", retention="6 months", format=logger_fmt, colorize=False, )
    logger.add( "logs/{time:YYYY-MM-DD}/g5.{time}.loguru.log.json", level="TRACE", retention="6 months", format=logger_fmt, colorize=False, serialize=True, )
    # fmt: on
except ModuleNotFoundError:
    import logging as logger

    logger.warning("Did not find `loguru` module, defaulting to `logging`")

DEFAULT_DATA_ROOT_DIR = Path("data/")


class _StreamToLogger:
    """Using this class to capture and log stderr and stdout output from subcommands.
    See: https://loguru.readthedocs.io/en/stable/resources/recipes.html#capturing-standard-stdout-stderr-and-warnings"""

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass

    def close(self):
        pass

    def isatty(self):
        pass


_stream_stdout = _StreamToLogger(level="SUCCESS")
_stream_stderr = _StreamToLogger(level="WARNING")


def _print_log(c_err: sarge.Capture, c_out: sarge.Capture, cmd_label: str):
    stderr_msg = c_err.readline()
    stdout_msg = c_out.readline()
    if len(stderr_msg) > 0:
        log_line = (cmd_label, "stderr", stderr_msg)
        print(log_line, file=sys.stderr)
    if len(stdout_msg) > 0:
        log_line = (cmd_label, "stdout", stdout_msg)
        print(log_line, file=sys.stdout)


def _run_cmd(
    cmd,
    cmd_label="",
    extra_wait_for_secs=0.1,
    # redirect_stderr=True, redirect_stdout=False # need to implement these flags in order to use this fn for hashdeep too
):
    logger.info(f"{cmd_label=}, running: {cmd=}")
    # use -1 for line-buffering
    # if redirect_stderr:
    c_err = sarge.Capture(buffer_size=1)
    # if redirect_stdout:
    c_out = sarge.Capture(buffer_size=1)

    p = sarge.run(
        cmd,
        async_=True,
        stdout=c_out,
        stderr=c_err,
    )

    # Check if subprocess has finished
    # https://stackoverflow.com/questions/35855263/check-if-subprocess-is-still-running-while-reading-its-output-line-by-line
    while p.poll_all()[-1] is None:
        # print(f'{p.poll_all()=}')
        _print_log(c_err, c_out, cmd_label)

    p.wait()  # seems to run the same without this
    logger.debug("Detected that the subprocess is ending...")

    logger.debug(
        f"We need to catch any output that we were too slow to catch before... continuing to monitor output for {extra_wait_for_secs=}"
    )
    # https://stackoverflow.com/questions/24374620/python-loop-to-run-for-certain-amount-of-seconds
    end_time = time.monotonic() + extra_wait_for_secs
    while time.monotonic() < end_time:
        _print_log(c_err, c_out, cmd_label)

    logger.debug(
        f"Command subprocess has exited: {cmd_label=}, {p.poll_all()[-1]=}, {p.returncodes=}"
    )
    for ret in p.returncodes:
        if ret != 0:
            logger.error(
                f"Warning: At least one of the returncodes is non-zero. The cmd might not have succeeded."
            )


def sha256sum(filepath):
    """https://www.quickprogrammingtips.com/python/how-to-calculate-sha256-hash-of-a-file-in-python.html"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def _hash_file_maybe(filepath, max_filesize: int = 500_000_000):
    # https://flaviocopes.com/python-get-file-details/
    full_digest, partial_digest = None, None
    if os.path.getsize(filepath) <= max_filesize:
        full_digest = sha256sum(filepath)
    else:
        logger.warning(
            f"not hashing full file because filesize exceeds {max_filesize=}"
        )
        # TBD implement partial digest of first XX bytes of large file
        partial_digest = None
    return {"full_digest": full_digest, "partial_digest": partial_digest}


def _file_info_digest(filepath):
    filepath_arg = filepath
    filepath_abs = Path(filepath).resolve()
    digest = _hash_file_maybe(filepath_abs)
    fileinfo = os.stat(filepath_abs)
    results = {
        "filepath_arg": filepath_arg,
        "filepath_abs": filepath_abs,
        "digest": digest,
        "fileinfo": fileinfo,
    }
    # logger.trace(results)
    return results


def _download_file_maybe(url: str, out_file_path=None):
    if out_file_path is None:
        out_file_path = Path(DEFAULT_DATA_ROOT_DIR) / Path(Path(url).name)

    if out_file_path.is_file():
        logger.warning(
            "Not downloading because there is already a file at: {} ({})".format(
                out_file_path, out_file_path.absolute()
            )
        )
    else:
        # https://stackoverflow.com/questions/1078524/how-to-specify-the-download-location-with-wget
        command = sarge.shell_format("wget -O {} {} ", out_file_path, url)
        # logger.info(f'Running {command=}')
        # sarge.run(command)
        _run_cmd(command, cmd_label="wget")


def setup_punkt():
    """Punkt tokenizer is used by the COHA preprocessing code (build_corpus...)"""
    import nltk

    nltk.download("punkt")


def _token_id(input_string: str) -> str:
    unique_hash_id = hashlib.sha256(str(input_string).encode()).hexdigest()
    return unique_hash_id[
        :5
    ]  # just a token ending, use it combined with timestamp to avoid collisions


def _freeze_hashdeep_rel(
    data_dirpath=DEFAULT_DATA_ROOT_DIR, hashdeep_outfile_name="data.hashdeep.rel.txt"
):
    command = ' hashdeep -c sha256 -revv -l "{}" > "{}" '
    command = sarge.shell_format(command, data_dirpath, hashdeep_outfile_name)
    logger.info(f"Running {command=}")
    sarge.run(command)


def _freeze_hashdeep_bare(
    data_dirpath=DEFAULT_DATA_ROOT_DIR, hashdeep_outfile_name="data.hashdeep.bare.txt"
):
    command = ' hashdeep -c sha256 -revv -b "{}" > "{}" '
    command = sarge.shell_format(command, data_dirpath, hashdeep_outfile_name)
    logger.info(f"Running {command=}")
    sarge.run(command)


def freeze_hashdeep(data_dirpath=DEFAULT_DATA_ROOT_DIR):
    _freeze_hashdeep_rel(data_dirpath)
    _freeze_hashdeep_bare(data_dirpath)


class coha(object):

    ZIP_EXTRACTION_DEST_DIR = Path(DEFAULT_DATA_ROOT_DIR) / "text.zip.coha.d/"
    SLICE_EXTRACTION_DEST_DIR_ROOT = Path(DEFAULT_DATA_ROOT_DIR) / "coha"

    @staticmethod
    def download_sample():
        """example CLI invocation: python g5_tools.py coha download_sample"""
        url = "https://www.corpusdata.org/coha/samples/text.zip"
        _download_file_maybe(url)

    @classmethod
    def _unzip_dataset(
        cls, exdir=None, zipfile_path=Path(DEFAULT_DATA_ROOT_DIR) / Path("text.zip")
    ):
        if exdir is None:
            exdir = cls.ZIP_EXTRACTION_DEST_DIR
        try:
            exdir.mkdir(exist_ok=False, parents=True)
        except:
            logger.warning(f"Aborting because {exdir=} ...seems to already exist")
            sys.exit(1)
        command = sarge.shell_format("unzip -q -d {} {} ", exdir, zipfile_path)
        sarge.run(command)

    @classmethod
    def clear(cls):
        """clear_all_except_zip. and also does not clear the output of build_corpus"""
        for e in [cls.ZIP_EXTRACTION_DEST_DIR, cls.SLICE_EXTRACTION_DEST_DIR_ROOT]:
            logger.info(f"Removing: {e}")
            shutil.rmtree(e, ignore_errors=True)

    @classmethod
    def extract_one_slice(
        cls,
        slice_pattern: str = "1910",
    ):
        """Example equivalent operation: mv data/text.zip.coha.d/*1910*.txt data/coha/1910/"""
        # dir where full dataset was unzipped to
        exdir = cls.ZIP_EXTRACTION_DEST_DIR

        slice_pattern = str(slice_pattern)
        logger.info(f"{slice_pattern=}")

        # dir where we will store our extracted slices
        slice_out_dir = cls.SLICE_EXTRACTION_DEST_DIR_ROOT / (slice_pattern + "/")
        slice_out_dir.mkdir(parents=True)

        found_matches = False
        fpaths = exdir.glob(("*_" + slice_pattern + "_*.txt"))
        for fpath in fpaths:
            found_matches = True  # if we found at least one match for the pattern
            # NOTE: mv is faster than cp, but harms reproducibility/idempotency
            command = sarge.shell_format("mv {} {}", fpath, slice_out_dir)
            # logger.debug(f'{command=}')
            # sarge.run(command)
            _run_cmd(command)

        if not found_matches:
            logger.warning("No matching files")

    @classmethod
    def _extract_several_slices(cls, *slices_patterns):
        logger.debug(f"{slices_patterns=}")
        for patt in slices_patterns:
            cls.extract_one_slice(slice_pattern=patt)

    @classmethod
    def extract(cls, *slices_patterns):
        logger.debug(f"{slices_patterns=}")
        cls._unzip_dataset()
        cls._extract_several_slices(*slices_patterns)

    @staticmethod
    def build_corpus(
        *dirpaths_for_input_slices,
        data_root_dir=DEFAULT_DATA_ROOT_DIR,
        do_txt: bool = True,
        do_json: bool = True,
    ):
        """Wrapper alternative to simplify the argparse version in the original build_coha_corpus.py."""
        logger.debug(f"{do_txt=}, {do_json=}")

        if __name__ == "__main__":
            setup_punkt()

        # this import needs punkt already downloaded in order to succeed
        from build_coha_corpus import build_train_test, build_data_sets

        data_root_dir = Path(data_root_dir)
        output_dir = data_root_dir / Path("outputs")

        input_folders = [Path(dirpath) for dirpath in dirpaths_for_input_slices]

        corpus_slice_labels = [
            str(Path(dirpath).name) for dirpath in dirpaths_for_input_slices
        ]

        paths_for_lm_output_train = [
            output_dir / Path(l) / Path("train.txt") for l in corpus_slice_labels
        ]
        paths_for_lm_output_test = [
            output_dir / Path(l) / Path("test.txt") for l in corpus_slice_labels
        ]

        json_output_files = [
            output_dir / Path(l) / Path("full_text.json.txt")
            for l in corpus_slice_labels
        ]

        logger.debug("Working with the following paths...")
        for el in [
            dirpaths_for_input_slices,
            input_folders,
            paths_for_lm_output_train,
            paths_for_lm_output_test,
            json_output_files,
        ]:
            logger.debug(el)

        logger.info("Ok, making changes to filesystem...")
        output_dir.mkdir(exist_ok=True)  # create outputs dir if it does not exist yet

        # Json generation runs much faster than txt, so we will do it first
        if do_json:
            # This function outputs json files. For COHA, these files are what
            # get_embeddings_scalable.get_slice_embeddings fn expects.
            logger.info("Running `build_data_sets()` to make json files...")
            build_data_sets(input_folders, json_output_files)

        if do_txt:
            logger.info(
                "Running the loop for `build_train_test()` to make train.txt and test.txt files..."
            )
            for infolder, lm_output_train, lm_output_test in zip(
                input_folders, paths_for_lm_output_train, paths_for_lm_output_test
            ):
                logger.info(f"{infolder=}")
                logger.debug(f"{lm_output_train=}, {lm_output_test=}")
                lm_output_train.parent.mkdir(exist_ok=True, parents=True)
                lm_output_test.parent.mkdir(exist_ok=True)
                build_train_test([infolder], lm_output_train, lm_output_test)

    @classmethod
    def prep_slices(cls, *slices):
        """This functions wraps the other ones in this class for easy usage. It will
        perform all-in-one: download sample, clear some data workspace, extract the
        slices, build the corpuses from the slices, cleanup afterwards.

        """
        logger.debug(f"{slices=}")
        slices = [str(l) for l in slices]
        logger.info(f"{slices=}")
        slice_dirs = [cls.SLICE_EXTRACTION_DEST_DIR_ROOT / l for l in slices]
        logger.debug(f"{slice_dirs=}")

        cls.download_sample()
        cls.clear()
        cls.extract(*slices)
        cls.build_corpus(*slice_dirs)
        cls.clear()


# might as well hardcode this since the list is so short
# fmt: off
SEMEVAL_WORDLIST = [ "attack", "bag", "ball", "bit", "chairman", "circle",
                     "contemplation", "donkey", "edge", "face", "fiction", "gas", "graft",
                     "head", "land", "lane", "lass", "multitude", "ounce", "part", "pin",
                     "plane", "player", "prop", "quilt", "rag", "record", "relationship",
                     "risk", "savage", "stab", "stroke", "thump", "tip", "tree", "twist",
                     "word", ]
# fmt: on

TESTING_BERT_TRAINTXT_PATH = "data/outputs/1910/train.txt"
TESTING_BERT_TESTTXT_PATH = "data/outputs/1910/test.txt"
TESTING_BERT_FULLTEXTJSON_PATH = "data/outputs/1910/full_text.json.txt"


class bert(object):
    @staticmethod
    def download_model(
        url="https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
    ):
        """NOTE: don't use this for now. Instead code is using the model that hugging face libraries will download automagically"""
        _download_file_maybe(url)

    @staticmethod
    def prep_coha(*slices):
        if len(slices) == 0:
            slices = list(
                range(1910, 2010)
            )  # Before 1900, the COHA sample corpus coverage is spotty

        coha.prep_slices(*slices)

        # coha.download_sample()
        # coha.clear()
        # coha.extract('1910', '1950')
        # coha.build_corpus("data/coha/1910/","data/coha/1950/")
        # coha.clear()

    @staticmethod
    def train(
        out="data/outputs/bert_training/",
        train=TESTING_BERT_TRAINTXT_PATH,
        test=TESTING_BERT_TESTTXT_PATH,
        batchSize=7,
        epochs=5.0,  # 5.0 is the default they use
        # **kwargs, # TBD: figure out how to implement generic kwargs passthru
    ):
        """Wraps `fine-tune_BERT.py` in order to run a test training.

        Original CLI invocation that this function wraps, as explained in the main README:

        python fine-tune_BERT.py --train_data_file pathToLMTrainSet --output_dir pathToOutputModelDir --eval_data_file pathToLMTestSet --model_name_or_path modelForSpecificLanguage --mlm --do_train --do_eval --evaluate_during_training
        """

        command = """python fine-tune_BERT.py \
        --train_data_file {pathToLMTrainSet} \
        --output_dir {pathToOutputModelDir} \
        --eval_data_file {pathToLMTestSet} \
        --model_name_or_path {modelForSpecificLanguage} \
        --mlm --do_train --do_eval --evaluate_during_training \
        \
        --per_gpu_train_batch_size {batchSize} \
        --per_gpu_eval_batch_size {batchSize} \
        --num_train_epochs {epochs}
        """
        # --config_name {pathToCfg}

        cmd = sarge.shell_format(
            command,
            pathToLMTrainSet=train,
            pathToOutputModelDir=out,
            pathToLMTestSet=test,
            modelForSpecificLanguage="bert-base-multilingual-cased",  # this works, it will download it from internet
            batchSize=batchSize,
            epochs=epochs,
        )
        # Here are other various notes from when i tried to make it work via loading a local, previously-downloaded model...

        # NOTE: '~' did not seem to work, absolute path needed
        # modelForSpecificLanguage = '/home/user/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_model.ckpt.data-00000-of-00001',

        # pathToCfg = '/home/user/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_config.json',

        # Also tried (with partial success):
        # ln -s ~/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/bert_config.json ~/projlogiciel/scalable_semantic_shift/data/multi_cased_L-12_H-768_A-12/config.json

        logger.debug(f"{cmd=}")
        _run_cmd(cmd, cmd_label="fine-tune_BERT.py", extra_wait_for_secs=5)

    @staticmethod
    def _get_gulordava_dict():
        """between get_embeddings_scalable.py and get_embeddings_scalable_semeval.py, they have a lot in common but seem to have two functions which are renamed/equivalent:

        get_embeddings_scalable.py has get_shifts(), get_slice_embeddings()

        get_embeddings_scalable_semeval.py has get_targets(), get_time_embeddings()

        get_targets() is designed to take the targets file from semeval zip, and has logic to strip the POS suffixes (_nn etc) from the list of words.

        whereas get_shifts() expects the Gulordava csv file, and gets its list of words from there.
        """
        # from get_embeddings_scalable import get_shifts # Gulordava csv
        # from get_embeddings_scalable_semeval import get_targets # SemEval targets file

        # wget https://marcobaroni.org/PublicData/gulordava_GEMS_evaluation_dataset.csv
        GULORDAVA_FILEPATH = (
            DEFAULT_DATA_ROOT_DIR / "gulordava_GEMS_evaluation_dataset.csv"
        ).resolve()
        logger.info(f"{GULORDAVA_FILEPATH=}")

        # shifts_dict = get_shifts(GULORDAVA_FILEPATH)
        """get_embeddings_scalable.get_shifts() is not compatible with this upstream version of the file; it seems(?) to expect format like "3 users" (mean_score:int word:str)"""

        shifts_dict = {}
        df_shifts = pd.read_csv(GULORDAVA_FILEPATH, sep=",", encoding="utf8")
        for idx, row in df_shifts.iterrows():
            ratings = [row[l] for l in ["p1", "p2", "p3", "p4", "p5"]]
            shifts_dict[row[0]] = mean(ratings)
        return shifts_dict

    @classmethod
    def _get_wordlist_for_extract_query(cls) -> list:
        gulordava_wordlist = list(cls._get_gulordava_dict().keys())

        wordlist = sorted(list(set(SEMEVAL_WORDLIST + gulordava_wordlist)))
        # wordlist = wordlist[:3] # for testing

        logger.info(f"{len(wordlist)=}, {wordlist=}")
        return wordlist

    @staticmethod
    def _make_mockup_dict_from_wordlist(wordlist: list) -> dict:
        """Should output something in this format: shifts_dict = {'cat': '42', 'dog': '42', 'bird': '42', 'house': '42', }

        Based on input like: wordlist = ['cat', 'dog', 'bird', 'house']

        """

        """This shifts_dict setup is necessary because the original function expects a file containing tokens and frequencies(?).

        Default path listed in `get_embeddings_scalable.py` is like so:

        parser.add_argument("--target_path", default='data/coha/Gulordava_word_meaning_change_evaluation_dataset.csv', type=str, help="Path to target file")

        This file does not seem to be readily available online, and in any case we don't necessarily want to restrict the set of words we look at in this same way.

        The functions in `get_embeddings_scalable.py` don't seem to actually do anything with the frequency info, so I have just included dummy values to fit the datastructure format.

        """
        mockup_dict = {}
        for w in wordlist:
            mockup_dict[w] = 42
        return mockup_dict

    @classmethod
    def _get_mockup_dict_for_extract_query(cls):
        mockup_dict = cls._make_mockup_dict_from_wordlist(
            cls._get_wordlist_for_extract_query()
        )
        logger.debug(f"{mockup_dict=}")
        return mockup_dict

    @classmethod
    def extract(
        cls,
        pathToFineTunedModel: str = "data/RESULTS_train_bert_coha/1910/pytorch_model.bin",
        dataset: str = "data/outputs/1910/full_text.json.txt",
        embeddings_path=None,
	target_path: str = "data/semeval2020_ulscd_eng/targets.txt",
	language: str = "english",
        gpu=True,
    ):
        """
                Wraps get_embeddings_scalable.py.
                Extract embeddings from the preprocessed corpus in .txt for corpus.

                The original repo's functions are designed for a single monomodel, in which multiple slices/corpuses are present and can be extracted.

                From original:

        python get_embeddings_scalable.py --corpus_paths pathToPreprocessedCorpusSlicesSeparatedBy';' --corpus_slices nameOfCorpusSlicesSeparatedBy';' --target_path pathToTargetFile --task chooseBetween'coha','durel','aylien' --path_to_fine_tuned_model pathToFineTunedModel --embeddings_path pathToOutputEmbeddingFile

        This creates a pickled file containing all contextual embeddings for all target words.
        """
        logger.info("Working with the following input parameters...")
        logger.info(f"{pathToFineTunedModel=}")
        logger.info(f"{dataset=}")
        logger.info(f"{gpu=}")

        import torch
        from transformers import BertTokenizer, BertModel
        from get_embeddings_scalable import get_slice_embeddings,get_shifts
        from get_embeddings_scalable_semeval import get_targets
        batch_size = 16
        max_length = 256
        logger.debug(f"hardcoded: {batch_size=}, {max_length=}")

        # slices = args.corpus_slices.split(';')

        # lang = 'English'
        task = "coha"

        # task = args.task

        # tasks = ['coha', 'aylien', 'durel']
        # if task not in tasks:
        #     print("Task not valid, valid choices are: ", ", ".join(tasks))
        #     sys.exit()

        # datasets = args.corpus_paths.split(';')

        datasets = [dataset]
        slices = [str(Path(dataset).parent.stem)]

        logger.info(f"{datasets=} ; {slices=}")

        """# embeddings_path: is path to output the embeddings file"""
        if embeddings_path is None:
            # dt = dt.format('YYYY-MM-DD HH:mm:ss')
            dt = pendulum.now().format("HH\hmm")
            stamp = dt + "_" + _token_id(pathToFineTunedModel + dataset)
            embeddings_path = (
                DEFAULT_DATA_ROOT_DIR / "embeddings" / stamp / f"{slices[0]}.pickle"
            )
        embeddings_path = embeddings_path.resolve()
        logger.info(f"We will output embeddings to file: {embeddings_path=}")

        """
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
        """

        # if len(args.path_to_fine_tuned_model) > 0:
        #     fine_tuned = True
        # else:
        #     fine_tuned = False

        fine_tuned = True

        # datasets = args.corpus_paths.split(';')

        # if len(args.path_to_fine_tuned_model) > 0:
        #     state_dict =  torch.load(args.path_to_fine_tuned_model)

        if task == "coha":
            lang = language
            shifts_dict = get_targets(target_path, lang)
            #shifts_dict = cls._get_mockup_dict_for_extract_query()

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
        # torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if gpu:
            pass
        else:
            torch_device = torch.device("cpu")
            logger.info(f"{torch_device=}")

        modelForSpecificLanguage = "bert-base-multilingual-cased"
        logger.info(f"{modelForSpecificLanguage=}")

        tokenizer = BertTokenizer.from_pretrained(
            modelForSpecificLanguage, do_lower_case=False
        )

        logger.debug(
            f"We will load state_dict from this file: {_file_info_digest(pathToFineTunedModel)=}"
        )
        logger.debug("Loading state_dict...")
        if gpu:
            state_dict = torch.load(pathToFineTunedModel)
        else:
            state_dict = torch.load(pathToFineTunedModel, map_location=torch_device)

        logger.debug("Now loading state_dict into a model...")
        model = BertModel.from_pretrained(
            modelForSpecificLanguage, state_dict=state_dict, output_hidden_states=True
        )
        logger.debug("state_dict has been loaded into a model")

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

        for dataset in datasets:
            logger.debug(f"{_file_info_digest(dataset)=}")

        logger.debug("Parameters that we will pass to get_slice_embeddings()...")
        logger.debug(
            f"{embeddings_path=}, {datasets=}, {tokenizer=}, (`model` too verbose to log here), {batch_size=}, {max_length=}, {lang=}, {shifts_dict=}, {task=}, {slices=}, {gpu=}"
        )
        logger.debug(f"{model=}")

        logger.info("Now running get_slice_embeddings()...")
        embeddings_path.parent.mkdir(exist_ok=True)
        embeddings_path = str(embeddings_path)
        # with contextlib.redirect_stdout(stream_stdout), contextlib.redirect_stderr(stream_stderr):
        get_slice_embeddings(
            embeddings_path,
            datasets,
            tokenizer,
            model,
            batch_size,
            max_length,
            lang,
            shifts_dict,
            task,
            slices,
            gpu=gpu,
        )

        logger.info(f"Output embeddings pickle should now be at: {embeddings_path=}")
        logger.debug(f"{_file_info_digest(embeddings_path)=}")

    @staticmethod
    def measure():
        """
        Wraps measure_semantic_shift.py.

        Method WD or JSD: "Wasserstein distance or Jensen-Shannon divergence".
        """
        # TBD implement this.
        # currently this step is instead implemented elsewhere, in measure_semantic_shift_merged.py
        pass


class eval(object):
    @staticmethod
    def _filter_results_file_with_wordlist(
        results_filepath: str, filter_list: list = SEMEVAL_WORDLIST
    ):
        results_filepath = str(Path(results_filepath).resolve())
        logger.info(f"Reading results from {results_filepath=}")
        df = pd.read_table(results_filepath, sep=";", encoding="utf8")

        df = df[df["word"].isin(filter_list)]
        df = df.sort_values("word")

        logger.debug(f"Filtered dataframe...:\n{df.head(999)}")
        return df

    @classmethod
    def filter_results(
        cls, results_filepath: str, filter_type: str = "semeval", out_filepath=""
    ):
        if filter_type == "semeval":
            filter_list = SEMEVAL_WORDLIST
        elif filter_type == "gulordava":
            filter_list = gulordava_wordlist = list(bert._get_gulordava_dict().keys())
        else:
            raise NotImplementedError(f"{filter_type=}")

        results_filepath = str(Path(results_filepath).resolve())
        df = cls._filter_results_file_with_wordlist(
            results_filepath, filter_list=filter_list
        )

        if out_filepath == "":
            outfile_name = (
                Path(results_filepath).stem + f".filtered_for_{filter_type}" + f".csv"
            )
            logger.debug(f"{outfile_name=}")
            out_filepath = Path(results_filepath).parent / outfile_name

        out_filepath = Path(out_filepath).resolve()
        logger.info(f"Saving filtered results into {out_filepath=}")
        df.to_csv(out_filepath, sep=";", encoding="utf-8", index=False)


class run(object):
    """# TBD implement these all strung together (skip the first step build_semeval_lm_train_test.py)

        # NOTE: additional step here is missing from list: producing 'data/cohasem_corpus2/full_text.json.txt'...

        # This is an example of the pipeline as run manually:

    time python build_semeval_lm_train_test.py  --corpus_paths data/semeval2020_ulscd_eng/corpus2/token/ccoha2.txt --target_path data/semeval2020_ulscd_eng/targets.txt --language english --lm_train_test_folder data/c2_corpus_out

    time python g5_tools.py bert train --train 'data/c2_corpus_out/train.txt' --out 'data/outputs/bert_c2_v2/' --test 'data/c2_corpus_out/test.txt'  --epochs 5 --batchSize 7

    time python g5_tools.py bert extract --pathToFineTunedModel 'data/outputs/bert_c2_v2/pytorch_model.bin' --dataset 'data/cohasem_corpus2/full_text.json.txt' --gpu True

    time python measure_semantic_shift_merged.py --embeddings_path 'data/embeddings/from_bert_v2/' --corpus_slices 'cohasem_corpus1;cohasem_corpus2'

    time python g5_tools.py eval filter_results 'results_coha/word_ranking_results_WD.csv' --filter_type semeval

    time python evaluate.py --task 'semeval' --gold_standard_path 'data/semeval2020_ulscd_eng/truth/graded.txt' --results_path 'results_coha/word_ranking_results_WD.filtered_for_semeval.csv' --corpus_slices 'cohasem_corpus1;cohasem_corpus2'"""

    @staticmethod
    def train_extract(
        train=TESTING_BERT_TRAINTXT_PATH,  # 'data/c2_corpus_out/train.txt',
        test=TESTING_BERT_TESTTXT_PATH,  # 'data/c2_corpus_out/test.txt',
        full_text_json=TESTING_BERT_FULLTEXTJSON_PATH,  # 'data/cohasem_corpus2/full_text.json.txt',
        epochs=5,
        batchSize=7,
    ):
        """NOTE: The default path arguments are just for testing purposes."""

        unattended_run_dirpath = (
            DEFAULT_DATA_ROOT_DIR / "unattended_runs" / str(pendulum.now())
        ).resolve()
        logger.info(f"{unattended_run_dirpath=}")

        # time python g5_tools.py bert train --train 'data/c2_corpus_out/train.txt' --out 'data/outputs/bert_c2_v2/' --test 'data/c2_corpus_out/test.txt'  --epochs 5 --batchSize 7

        train_output_dir = (unattended_run_dirpath / "bert_train").resolve()

        unattended_run_dirpath.parent.mkdir(exist_ok=True)
        unattended_run_dirpath.mkdir(exist_ok=False)
        train_output_dir.mkdir(exist_ok=False)

        logger.debug(f"{os.stat(unattended_run_dirpath)=}")
        logger.debug(f'{ [p for p in unattended_run_dirpath.rglob("*")] =}')

        bert.train(
            train=train,
            test=test,
            out=str(train_output_dir),
            epochs=epochs,
            batchSize=batchSize,
        )

        logger.debug(f"{os.stat(unattended_run_dirpath)=}")
        logger.debug(f'{ [p for p in unattended_run_dirpath.rglob("*")] =}')

        # time python g5_tools.py bert extract --pathToFineTunedModel 'data/outputs/bert_c2_v2/pytorch_model.bin' --dataset 'data/cohasem_corpus2/full_text.json.txt' --gpu True

        dataset = Path(full_text_json).resolve()
        slice = str(Path(dataset).parent.stem)
        embeddings_out_filepath = (unattended_run_dirpath / f"{slice}.pickle").resolve()

        bert.extract(
            pathToFineTunedModel=str(train_output_dir / "pytorch_model.bin"),
            dataset=full_text_json,
            gpu=True,
            embeddings_path=embeddings_out_filepath,
        )
        logger.debug(f"{os.stat(unattended_run_dirpath)=}")
        logger.debug(f'{ [p for p in unattended_run_dirpath.rglob("*")] =}')


@logger.catch
def main():
    logger.debug(f"CLI invocation argument list: {sys.argv=}")
    with contextlib.redirect_stdout(_stream_stdout), contextlib.redirect_stderr(
        _stream_stderr
    ):
        print("NOTE: Standard output (stdout) is sent to added handlers, like so.")
        print(
            "NOTE: Standard error  (stderr) is sent to added handlers, like so.",
            file=sys.stderr,
        )
        # Note that with Fire in here, fire can never print to stdout (this
        # even messes up the Fire autogenerated help pages). If we ever decide
        # we have a legitimate reason to print to stdout (for example we want
        # to redirect to a file with >), then we can move the context mgr to
        # each spot where we call out to an external fn/cmd, rather than having
        # it centralized here.
        fire.Fire()
        # The upside of doing this centrally here is that we will be able to
        # catch *all* the output that any subcommands try to send, and log it
        # for reference.
    logger.debug("end of script")


if __name__ == "__main__":
    sys.exit(main())
