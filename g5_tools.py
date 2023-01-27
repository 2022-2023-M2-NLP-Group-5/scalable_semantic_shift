#!/usr/bin/env python3
"""Collection of tools for our purposes.

Example prep for new node/cluster (if you're running on a cluster that is
already set up, then you just need some of these commands):

oarsub -I -p "cluster='gemini'" -l gpu=1,walltime=1:00  -t exotic  # lyon

# best clusters for gpus on nancy: grue, gruss and graffiti

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

import contextlib
import hashlib
import os
import shutil
import sys
import time
from collections import namedtuple
from pathlib import Path

import filehash
import fire
import pandas as pd
import pendulum
import sarge
import torch
from numpy import mean
from transformers import BertModel, BertTokenizer

filehasher = filehash.FileHash()

if __name__ == "__main__":
    try:
        from loguru import logger

        logger.remove()
        # https://loguru.readthedocs.io/en/stable/api/logger.html?#loguru._logger.Logger.add
        LOGGER_FORMAT = "<green>{time:MM-DD HH:mm:ss.SSS}</green>|<level>{elapsed}</level>|<level>{level: <8}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        # dt = dt.format('YYYY-MM-DD HH:mm:ss')
        timestamp = pendulum.now()  # .format()
        logfile = f"g5.{timestamp}.loguru"
        # fmt: off
        logger.add( sys.stderr, backtrace=True, diagnose=True, format=LOGGER_FORMAT )
        logger.add( "logs/{time:YYYY-MM-DD}/{time:HH}h/" + logfile + ".log", retention="6 months", format=LOGGER_FORMAT, colorize=True, )
        logger.add( "logs/{time:YYYY-MM-DD}/{time:HH}h/" + logfile + ".txt",  retention="6 months", format=LOGGER_FORMAT, colorize=False, )
        logger.add( "logs/{time:YYYY-MM-DD}/{time:HH}h/" + logfile + ".json", retention="6 months", format=LOGGER_FORMAT, colorize=False, serialize=True, )
        # fmt: on
    except ModuleNotFoundError:
        import logging as logger

        logger.warning("Did not find `loguru` module, defaulting to `logging`")

DEFAULT_DATA_ROOT_DIR = Path("data/")


class _StreamToLogger:
    """Using this class to capture and log stderr and stdout output from subcommands.
    See: https://loguru.readthedocs.io/en/stable/resources/recipes.html#capturing-standard-stdout-stderr-and-warnings"""

    def __init__(self, level="INFO", prefix=""):
        self._level = level
        self._prefix = prefix

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, str(self._prefix) + line.rstrip())

    def flush(self):
        pass

    def close(self):
        pass

    def isatty(self):
        pass


_stream_stdout = _StreamToLogger(level="INFO", prefix="[STDOUT] ")
_stream_stderr = _StreamToLogger(level="INFO", prefix="[STDERR] ")
_redirect_stdout = contextlib.redirect_stdout(_stream_stdout)
_redirect_stderr = contextlib.redirect_stderr(_stream_stderr)


def _write_list_to_file(input_list, outfile_path=sys.stdout):
    """Write a list to a file, newline-separated, each element of list on its own line."""
    outfile_path = Path(outfile_path).resolve()
    with open(outfile_path, "w", encoding="utf8") as f:
        f.writelines(
            "\n".join(input_list)
        )  # writelines() does not append its own newlines
    # logger.debug(f"file is now at {outfile_path=}")


def _print_log(c_err: sarge.Capture, c_out: sarge.Capture, cmd_label: str):
    stderr_msg = c_err.readline()
    stdout_msg = c_out.readline()
    if len(stderr_msg) > 0:
        StdErr = namedtuple("StdErr", ["l", "m"])
        log_line = StdErr(l=cmd_label, m=stderr_msg)
        logger.info(f"{log_line}")
    if len(stdout_msg) > 0:
        StdOut = namedtuple("StdOut", ["l", "m"])
        log_line = StdOut(l=cmd_label, m=stdout_msg)
        logger.info(f"{log_line}")


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
    # logger.debug("Detected that the subprocess is ending...")

    if extra_wait_for_secs > 0.01:
        logger.debug(
            f"We need to catch any output that we were too slow to catch before... continuing to monitor output for {extra_wait_for_secs=}"
        )
    end_time = time.monotonic() + extra_wait_for_secs
    # This while loop is outside the if clause on purpose, otherwise we may
    # miss near-instantaneous output like from git.
    # https://stackoverflow.com/questions/24374620/python-loop-to-run-for-certain-amount-of-seconds
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


def reduce_corpus(
    corpus_filepath="data/semeval2020_ulscd_eng/corpus1/token/ccoha1.txt",
    target_filepath="data/wordlists/synonyms/no_mwe/bag.txt",
    output_filepath="data/syn/c1_AUTOTEST/c1_EN_reduced.txt",
    lang="english",
):
    """
    python reduce_corpus.py --corpus_filepath 'data/semeval2020_ulscd_eng/corpus1/token/ccoha1.txt' --target_filepath 'data/wordlists/synonyms/no_mwe/bag.txt' --output_filepath 'data/syn/c1/c1_EN_reduced.txt' --lang english"""
    cmd = (
        "python reduce_corpus.py  "
        "--corpus_filepath {}  "
        "--target_filepath {}  "
        "--output_filepath {}  "
        "--lang {}  "
    )
    cmd = sarge.shell_format(
        cmd, corpus_filepath, target_filepath, output_filepath, lang
    )
    _run_cmd(cmd, cmd_label=cmd.split(" ")[1])


def sha256sum(filepath):
    """https://www.quickprogrammingtips.com/python/how-to-calculate-sha256-hash-of-a-file-in-python.html"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def _hash_file_maybe(
    filepath,
    max_filesize: int = 1_000_000_000,  # 1 gb
):
    """Calculate a checksum of a file unless the file is too big and it will take too long."""
    # 1gb takes ~1.9 seconds to hash on my system. The BERT model *.bin files are 682 mb.
    full_digest, partial_digest = None, None
    # how to get file size https://flaviocopes.com/python-get-file-details/
    if os.path.getsize(filepath) <= max_filesize:
        full_digest = sha256sum(filepath)
    else:
        logger.debug(f"not hashing full file because filesize exceeds {max_filesize=}")
        # TBD implement partial digest of first XX bytes of large file
        partial_digest = None
    return {"full_digest": full_digest, "partial_digest": partial_digest}


def _file_info_digest(filepath):
    filepath_arg = filepath
    filepath_abs = str(Path(filepath).resolve())
    digest = _hash_file_maybe(filepath_abs)
    fileinfo = os.stat(filepath_abs)
    results = {
        "digest": digest,
        "filepath_arg": filepath_arg,
        "filepath_abs": filepath_abs,
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
        _run_cmd(command, cmd_label="wget")

    logger.debug(f"{_file_info_digest(out_file_path)=}")


def setup_punkt():
    """Punkt tokenizer is used by the COHA preprocessing code (build_corpus...)"""
    with _redirect_stderr, _redirect_stdout:
        # stderr and stdout capture doesnt seem to work with nltk
        import nltk

        nltk.download("punkt")


def _token_id(items_to_hash) -> str:
    """each item is probably a path or other type of string"""
    # TODO hash also the contents of files?
    hashes_list = []
    for item in items_to_hash:
        hash = hashlib.sha256(str(item).encode()).hexdigest()
        hashes_list.append(hash)
    hashes_list = sorted(hashes_list)  # we want to sort, but we do NOT want to dedup
    hashes_summary = "".join(hashes_list)
    unique_hash_id = hashlib.sha256(str(hashes_summary).encode()).hexdigest()
    unique_hash_id = unique_hash_id[:6]
    # just a token ending, use it combined with timestamp to avoid collisions
    return unique_hash_id


def _stamp(items_to_hash, dt_format="MM-DD_HH:mm:ss"):
    dt = pendulum.now().format(dt_format)
    stamp = dt + "_p." + _token_id(items_to_hash)
    return stamp


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
            logger.error(f"Aborting because {exdir=} ...seems to already exist")
            sys.exit(1)
        command = sarge.shell_format("unzip -q -d {} {} ", exdir, zipfile_path)
        _run_cmd(command, cmd_label="unzip")

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
            _run_cmd(command, cmd_label="mv", extra_wait_for_secs=0)

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
        output_dir="",
        do_txt: bool = True,
        do_json: bool = True,
    ):
        """Wrapper alternative to simplify the argparse version in the original build_coha_corpus.py."""
        logger.debug(f"{do_txt=}, {do_json=}")

        # TODO add more file info digest into this fn. Also, display where the output paths are clearly in logs.

        if __name__ == "__main__":
            # Punkt tokenizer is used by build_coha_corpus.py
            setup_punkt()

        # this import needs punkt already downloaded in order to succeed
        from build_coha_corpus import build_data_sets, build_train_test

        # First we setup all our paths that we will use, without modifying anything on disk yet...

        if output_dir == "":
            output_dir = Path(data_root_dir) / "outputs" / "corpus"

        output_dir = Path(output_dir).resolve()

        dirpaths_for_input_slices = [
            Path(p).resolve() for p in dirpaths_for_input_slices
        ]

        # TODO collapse this into a single variable
        input_folders = dirpaths_for_input_slices

        # TODO this is too rigid, make this better (we have made slice labels manual elsewhere)
        corpus_slice_labels = [
            str(Path(dirpath).name) for dirpath in dirpaths_for_input_slices
        ]

        paths_for_lm_output_train = [
            (output_dir / l / "train.txt") for l in corpus_slice_labels
        ]
        paths_for_lm_output_test = [
            (output_dir / l / "test.txt") for l in corpus_slice_labels
        ]

        json_output_files = [
            (output_dir / l / "full_text.json.txt") for l in corpus_slice_labels
        ]

        logger.debug("Working with the following paths...")

        logger.debug(f"{dirpaths_for_input_slices=}")
        logger.debug(f"{input_folders=}")
        logger.debug(f"{paths_for_lm_output_train=}")
        logger.debug(f"{paths_for_lm_output_test=}")
        logger.debug(f"{json_output_files=}")

        logger.info(f"{output_dir=}")

        # for d in dirpaths_for_input_slices:
        #     logger.debug(f"{filehasher.hash_dir(d, pattern='*')=}")

        # Now we start doing actions that will modify things on disk...

        logger.info("Ok, making changes to files...")
        output_dir.mkdir(exist_ok=True)  # create outputs dir if it does not exist yet

        # Json generation runs much faster than txt, so we will do it first
        if do_json:
            for json_filepath in json_output_files:
                json_filepath.parent.mkdir(exist_ok=True, parents=True)
            # This function outputs json files. For COHA, these files are what
            # get_embeddings_scalable.get_slice_embeddings fn expects.
            logger.info("Running `build_data_sets()` to make json files...")
            with _redirect_stdout, _redirect_stderr:
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
                with _redirect_stdout, _redirect_stderr:
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
    def _read_wordlist_from_file(input_path) -> list:
        """This fn expects a text file with one word per line. It will work on
        data/semeval2020_ulscd_ger/targets.txt ; but not on the English semeval target
        file (English one is different format, it includes POS suffixes).
        """
        targets_list = []
        with open(input_path, "r", encoding="utf8") as f:
            for line in f:
                target = line.strip()
                targets_list.append(target)
        logger.debug(f"{len(targets_list)=}")
        wordlist = sorted(set(targets_list))
        for w in wordlist:
            if "_" in w:
                logger.error(
                    "Detected an underscore inside a word. "
                    "Did you run this on SemEval Eng targets.txt? "
                    "Instead you must first use `make_wordlistfile_from_semeval_eng_targets_txt()` to create a wordlist.txt."
                )
        return wordlist

    @staticmethod
    def make_wordlistfile_from_semeval_eng_targets_txt(
        targets_txt_path="data/semeval2020_ulscd_eng/targets.txt",
        outfile_path="data/semeval2020_ulscd_eng/wordlist.txt",
    ):
        logger.debug(f"{targets_txt_path=}")
        targets_list = []
        with open(targets_txt_path, "r", encoding="utf8") as f:
            for line in f:
                target = line.strip()
                target_no_pos = target[:-3]  # remove POS endings like: _nn
                targets_list.append(target_no_pos)
        outfile_path = Path(outfile_path).resolve()
        with open(outfile_path, "w", encoding="utf8") as f:
            f.writelines(
                "\n".join(targets_list)
            )  # writelines() does not append its own newlines
        logger.info(f"Semeval english wordlist file is now at {outfile_path=}")

    @staticmethod
    def _get_gulordava_dict():
        """between get_embeddings_scalable.py and get_embeddings_scalable_semeval.py, they have a lot in common but seem to have two functions which are renamed/equivalent:

        get_embeddings_scalable.py has get_shifts(), get_slice_embeddings()

        get_embeddings_scalable_semeval.py has get_targets(), get_time_embeddings()

        get_targets() is designed to take the targets file from semeval zip, and has logic to strip the POS suffixes (_nn etc) from the list of words.

        whereas get_shifts() expects the Gulordava csv file, and gets its list of words from there.
        """
        # from get_embeddings_scalable import get_shifts # Gulordava csv # NOTE: when used, wrap with contextlib
        # from get_embeddings_scalable_semeval import get_targets # SemEval targets file # NOTE: when used, wrap with contextlib

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
    def _get_wordlist_for_extract_query(
        cls, path="data/semeval2020_ulscd_ger/targets.txt"
    ) -> list:
        # gulordava_wordlist = list(cls._get_gulordava_dict().keys())
        # wordlist = SEMEVAL_WORDLIST + gulordava_wordlist
        logger.debug(f"{path=}")

        wordlist = cls._read_wordlist_from_file(input_path=path)

        wordlist = sorted(list(set(wordlist)))
        # wordlist = wordlist[:3]  # for testing

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

    # @classmethod
    # def _get_mockup_dict_for_extract_query(cls):
    #     mockup_dict = cls._make_mockup_dict_from_wordlist(
    #         cls._get_wordlist_for_extract_query()
    #     )
    #     logger.debug(f"{mockup_dict=}")
    #     return mockup_dict

    @classmethod
    def extract(
        cls,
        slice_label: str,
        lang: str,  # "English", # or "German" # upper-cased here (some other fns take lowercase)
        embeddings_path,  # ="",
        pathToFineTunedModel="data/averie_bert_training_c1/pytorch_model.bin",  # "data/RESULTS_train_bert_coha/1910/pytorch_model.bin",
        dataset="data/outputs/1910/full_text.json.txt",
        wordlist_path="data/semeval2020_ulscd_eng/wordlist.txt",  # "data/semeval2020_ulscd_ger/targets.txt",
        gpu=True,
        batch_size=16,
        max_length=256,
    ):
        """
                Wraps get_embeddings_scalable.py.
                Extract embeddings from the preprocessed corpus in .txt for corpus.
                This creates a pickled file containing all contextual embeddings for all target words.

                The original repo's functions are designed for a single monomodel, in which multiple slices/corpuses are present and can be extracted.

                From original:

        python get_embeddings_scalable.py --corpus_paths pathToPreprocessedCorpusSlicesSeparatedBy';' --corpus_slices nameOfCorpusSlicesSeparatedBy';' --target_path pathToTargetFile --task chooseBetween'coha','durel','aylien' --path_to_fine_tuned_model pathToFineTunedModel --embeddings_path pathToOutputEmbeddingFile

        """
        logger.info("Working with the following input parameters...")
        logger.warning(f"{embeddings_path=}")
        logger.info(f"{slice_label=}")
        logger.info(f"{lang=}")
        logger.info(f"{pathToFineTunedModel=}")
        logger.info(f"{dataset=}")
        logger.info(f"{wordlist_path=}")
        logger.info(f"{gpu=}")

        logger.debug(f"{batch_size=}, {max_length=}")

        logger.debug(f"{_file_info_digest(pathToFineTunedModel)=}")
        logger.debug(f"{_file_info_digest(dataset)=}")
        logger.debug(f"{_file_info_digest(wordlist_path)=}")

        from get_embeddings_scalable import get_slice_embeddings

        task = "coha"
        datasets = [dataset]
        slices = [slice_label]

        logger.info(f"{datasets=} ; {slices=}")

        """# embeddings_path: is path to output the embeddings file"""
        """
        if embeddings_path == "":
            stamp = _stamp(
                [pathToFineTunedModel, dataset]
            )  # TODO add other items into this list?
            # _slice_label = slice_label # slices[0]
            embeddings_path = (
                DEFAULT_DATA_ROOT_DIR / "embeddings" / stamp / f"{slice_label}.pickle"
            )
        """

        embeddings_path = Path(embeddings_path).resolve()
        logger.info(f"We will output embeddings to file: {embeddings_path=}")

        """
        NOTE: These dataset paths correspond to the files output by build_coha_corpus.build_data_sets (for coha, these are json format -- unclear why).
        The files are json format, but in the original SSS scripts, are not named .json, they are named simply .txt.

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

        fine_tuned = True

        """
        if task == "coha":
            # lang = "English" # commented out now, we are adding explicit parameter

        """

        # shifts_dict = get_shifts(args.target_path)
        # shifts_dict = cls._get_mockup_dict_for_extract_query()
        shifts_dict = cls._make_mockup_dict_from_wordlist(
            cls._get_wordlist_for_extract_query(path=wordlist_path)
        )
        logger.debug(f"{shifts_dict=}")

        # https://stackoverflow.com/questions/65882750/please-use-torch-load-with-map-location-torch-devicecpu
        # torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        """ # TODO moved this elsewhere, can delete soon
        if gpu:
            pass
        else:
            torch_device = torch.device("cpu")
            logger.info(f"{torch_device=}")
        """

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
            torch_device = torch.device("cpu")
            logger.info(f"{torch_device=}")
            state_dict = torch.load(pathToFineTunedModel, map_location=torch_device)

        logger.debug("Now loading state_dict into a model...")
        model = BertModel.from_pretrained(
            modelForSpecificLanguage, state_dict=state_dict, output_hidden_states=True
        )
        logger.debug("state_dict has been loaded into a model")

        if gpu:
            model.cuda()
        model.eval()

        if len(datasets) > 1:
            logger.warning(
                "This datasets list has more than one dataset in it, is this function built for that?"
            )
        for dataset in datasets:
            logger.debug(f"{_file_info_digest(dataset)=}")

        logger.debug("Parameters that we will pass to get_slice_embeddings()...")
        logger.debug(
            f"{embeddings_path=}, {datasets=}, {tokenizer=}, (`model` too verbose to log here), {batch_size=}, {max_length=}, {lang=}, {shifts_dict=}, {task=}, {slices=}, {gpu=}"
        )
        # logger.debug(f"{model=}")

        logger.info("Now running get_slice_embeddings()...")
        embeddings_path.parent.mkdir(exist_ok=True)
        embeddings_path = str(embeddings_path)
        with _redirect_stdout, _redirect_stderr:
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

    @classmethod
    def filter_dataset_and_extract(
        cls,
        slice_label: str,
        corpus_filepath="data/semeval2020_ulscd_eng/corpus1/token/ccoha1.txt",  # full, original source dataset, in a single txt file, to be filtered
        lang="english",
        pathToFineTunedModel="data/averie_bert_training_c1/pytorch_model.bin",  # "data/RESULTS_train_bert_coha/1910/pytorch_model.bin",
        # dataset="data/outputs/1910/full_text.json.txt", # leftover cruft from copying from another fn definition
        wordlist_path="data/wordlists/synonyms/no_mwe/bag.txt",  # "data/semeval2020_ulscd_eng/wordlist.txt",  # "data/semeval2020_ulscd_ger/targets.txt",
        # embeddings_path=None,  # tbd change this to empty string
        # gpu=True,
        # filter_dataset: bool = True,
        # batch_size=16,
        # max_length=256,
    ):
        logger.info(f"{corpus_filepath=}")
        logger.info(f"{lang=}")
        logger.info(f"{pathToFineTunedModel=}")
        logger.info(f"{wordlist_path=}")

        base_working_dirpath = (
            DEFAULT_DATA_ROOT_DIR
            / "unattended_runs"
            / _stamp([corpus_filepath, lang, pathToFineTunedModel, wordlist_path])
        )
        filtered_dataset_dirpath = base_working_dirpath / "filtered_corpus"

        embeddings_output_path = (
            base_working_dirpath / "embeddings" / f"{slice_label}.pickle"
        )
        logger.debug(f"{embeddings_output_path=}")

        filtered_dataset_txt_filepath = filtered_dataset_dirpath / "reduced.txt"
        dataset_json = filtered_dataset_dirpath / "full_text.json.txt"
        logger.debug(f"{dataset_json=}")

        base_working_dirpath.mkdir(exist_ok=False, parents=True)
        filtered_dataset_dirpath.mkdir(exist_ok=False, parents=True)

        logger.debug(f"{base_working_dirpath=}")
        logger.debug(f"{filtered_dataset_dirpath=}")
        logger.debug(f"{filtered_dataset_txt_filepath=}")

        logger.info("now reducing corpus")
        reduce_corpus(
            corpus_filepath=corpus_filepath,  # "data/semeval2020_ulscd_eng/corpus1/token/ccoha1.txt",
            target_filepath=wordlist_path,  # "data/wordlists/synonyms/no_mwe/bag.txt",
            output_filepath=filtered_dataset_txt_filepath,  # "data/syn/c1_AUTOTEST/c1_EN_reduced.txt",
            lang=lang.lower(),
        )

        logger.info("now running coha.build_corpus() to build json version")
        coha.build_corpus(
            filtered_dataset_dirpath,
            do_txt=False,
            do_json=True,
            output_dir=base_working_dirpath,
        )

        logger.info("now extracting pickle")
        dataset_json = filtered_dataset_dirpath / "full_text.json.txt"
        logger.debug(f"{dataset_json=}")
        embeddings_output_path = (
            base_working_dirpath / "embeddings" / f"{slice_label}.pickle"
        )
        logger.debug(f"{embeddings_output_path=}")
        cls.extract(
            slice_label=slice_label,
            lang=lang.capitalize(),
            pathToFineTunedModel=pathToFineTunedModel,
            dataset=dataset_json,
            wordlist_path=wordlist_path,
            embeddings_path=str(embeddings_output_path),
        )

    @classmethod
    def loop_extract(
        cls,
        list_of_slice_model_dataset_tuples: list,
        wordlist_path,
        lang: str = "english",
    ):
        # def loop_extract(cls, model_dataset_pairs_pathlist: list[tuple], *kwargs):
        # '''operate on paths'''
        for slice_label, model_path, dataset_path in list_of_slice_model_dataset_tuples:
            logger.info(
                f"Running extraction for: {slice_label=}, {model_path=}, {dataset_path=}"
            )
            cls.extract(
                slice_label=slice_label,
                lang=lang.capitalize(),  # extract takes capitalized, reduce_corpus takes lower (difference comes from the original SSS scripts)
                pathToFineTunedModel=model_path,  # "data/averie_bert_training_c1/pytorch_model.bin",
                dataset=dataset_path,  # "data/outputs/1910/full_text.json.txt",
                wordlist_path=wordlist_path,  # "data/semeval2020_ulscd_eng/wordlist.txt",
            )

            # embeddings_path=None,

            # *kwargs,

    @classmethod
    def test_le(cls):
        args = [
            # ("model", "dataset"),
            (
                "c1_test",
                "data/outputs/bert_en_de_c1/pytorch_model.bin",
                "data/outputs/c1/full_text.json.txt",
            ),
            (
                "c2_test",
                "data/outputs/bert_en_de_c2/pytorch_model.bin",
                "data/outputs/c2/full_text.json.txt",
            ),
        ]
        wordlist_path = "data/wordlists/synonyms/no_mwe/bag.txt"
        cls.loop_extract(args, wordlist_path=wordlist_path)

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
        results_filepath: str,
        filter_list: list = SEMEVAL_WORDLIST,  # TBD implement german wordlist/wordlist from a file
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
    ):  # TBD implement german semeval
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
        slice_label: str,
        lang: str = "english",
        train=TESTING_BERT_TRAINTXT_PATH,  # 'data/c2_corpus_out/train.txt',
        test=TESTING_BERT_TESTTXT_PATH,  # 'data/c2_corpus_out/test.txt',
        full_text_json=TESTING_BERT_FULLTEXTJSON_PATH,  # 'data/cohasem_corpus2/full_text.json.txt',
        epochs=5,
        batchSize=7,
    ):
        """NOTE: The default path arguments are just for testing purposes."""

        logger.debug(f"{_file_info_digest(train)=}")
        logger.debug(f"{_file_info_digest(test)=}")
        logger.debug(f"{_file_info_digest(full_text_json)=}")

        unattended_run_dirpath = (
            DEFAULT_DATA_ROOT_DIR / "unattended_runs" / str(pendulum.now())
        ).resolve()
        logger.info(f"{unattended_run_dirpath=}")

        # time python g5_tools.py bert train --train 'data/c2_corpus_out/train.txt' --out 'data/outputs/bert_c2_v2/' --test 'data/c2_corpus_out/test.txt'  --epochs 5 --batchSize 7

        train = Path(train).resolve()
        test = Path(test).resolve()
        train_output_dir = (unattended_run_dirpath / "bert_train").resolve()

        unattended_run_dirpath.parent.mkdir(exist_ok=True)
        unattended_run_dirpath.mkdir(exist_ok=False)
        train_output_dir.mkdir(exist_ok=False)

        logger.debug(f"{os.stat(unattended_run_dirpath)=}")
        logger.debug(f'{ [p for p in unattended_run_dirpath.rglob("*")] =}')

        bert.train(
            train=str(train),
            test=str(test),
            out=str(train_output_dir),
            epochs=epochs,
            batchSize=batchSize,
        )

        logger.debug(f"{os.stat(unattended_run_dirpath)=}")
        logger.debug(f'{ [p for p in unattended_run_dirpath.rglob("*")] =}')

        # time python g5_tools.py bert extract --pathToFineTunedModel 'data/outputs/bert_c2_v2/pytorch_model.bin' --dataset 'data/cohasem_corpus2/full_text.json.txt' --gpu True

        full_text_json = Path(full_text_json).resolve()
        # dataset = Path(full_text_json).resolve()
        slice = str(Path(full_text_json).parent.stem)
        embeddings_out_filepath = (unattended_run_dirpath / f"{slice}.pickle").resolve()

        bert.extract(
            slice_label=slice_label,
            lang=lang.capitalize(),  # extract takes capitalized, reduce_corpus takes lower (difference comes from the original SSS scripts)
            pathToFineTunedModel=str(train_output_dir / "pytorch_model.bin"),
            dataset=full_text_json,
            gpu=True,
            embeddings_path=embeddings_out_filepath,
        )
        logger.debug(f"{os.stat(unattended_run_dirpath)=}")
        logger.debug(f'{ [p for p in unattended_run_dirpath.rglob("*")] =}')
        logger.debug(f"{filehasher.hash_dir(unattended_run_dirpath, pattern='*')=}")


import itertools

from nltk.corpus import wordnet

# TBD make automatic
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class wn(object):
    """WordNet stuff

    currently only english synonyms implemented; TODO implement german synonyms"""

    def __init__(self, drop_mwe=True):
        self.drop_mwe = drop_mwe

    @staticmethod
    def _get_syn_ant(word):
        """get synonyms (and antonyms)

        this fn works in python 3.8 and 3.9"""
        synonyms = []
        antonyms = []

        for syn in wordnet.synsets(str(word)):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())

        synonyms = sorted(set(synonyms))
        antonyms = sorted(set(antonyms))

        return {"synonyms": synonyms, "antonyms": antonyms}

    def _syn_method1(self, word):
        """Get synoyms for a word."""
        syn = self._get_syn_ant(word)["synonyms"]
        synonyms = sorted(set(syn))
        if self.drop_mwe:
            synonyms = [el for el in synonyms if "_" not in el]
        synonyms = [el.replace("_", " ") for el in synonyms]
        return sorted(synonyms)

    def syn(self, *args, **kwargs):
        return self._syn_method1(*args, **kwargs)

    def make_syn_files(self, targets_file="", outfiles_dir=""):
        logger.debug(f"{_file_info_digest(targets_file)=}")
        wordlist = bert._read_wordlist_from_file(targets_file)
        outfiles_dir = Path(outfiles_dir)
        logger.debug(f"{self.drop_mwe=}")
        for w in wordlist:
            syn_list = self._syn_method1(w)
            outfile_path = (outfiles_dir / f"{w}.txt").resolve()
            logger.info(f"Writing: {outfile_path=}")
            _write_list_to_file(syn_list, outfile_path)


def _do_initial_logging():
    logger.debug(f"Our logs will be named ...{logfile}...")
    logger.debug(f"{LOGGER_FORMAT=}")
    logger.debug(f"CLI invocation argument list: {sys.argv=}")
    logger.debug(f"{sys.flags=}")
    GIT_CMDS = [
        "git rev-parse HEAD",
        "git show --no-patch --oneline",
        "git status --porcelain=v2 --branch -z",
    ]
    for cmd in GIT_CMDS:
        _run_cmd(cmd, cmd_label=cmd, extra_wait_for_secs=0)


@logger.catch
def _main():
    _do_initial_logging()
    fire.Fire()
    logger.debug("end of script")


if __name__ == "__main__":
    sys.exit(_main())
