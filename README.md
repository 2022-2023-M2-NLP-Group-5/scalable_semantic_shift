# Group 5 contributions and wrappers around scalable_semantic_shift codebase

This repository, forked from [the original](https://github.com/matejMartinc/scalable_semantic_shift), stores the code we use for our backend, including additions we have made to the codebase. Our main contributions are in the following files:

* [g5_tools.py](./g5_tools.py)
* [measure_semantic_shift_merged.py](measure_semantic_shift_merged.py)
* [measure_semantic_shift_visualisation.py](measure_semantic_shift_visualisation.py)
* [reduce_corpus.py](reduce_corpus.py)


## Content
- [Install instructions](#install-instructions)
- [Usage instructions](#usage-instructions)

## Install instructions

To setup the project, you have to:
1. clone the repository (either directly or thru the main super-repo where this repo is a submodule);
2. [install the Python dependencies](#python-dependencies);
3. [download the datasets](#).

### Python dependencies
It is recommended to have a dedicated environment for the project.
We recommend using mambaforge, but you can also use Pip to initiate this environment. 

#### Conda setup
Follow the following steps to setup the project with Anaconda. 
We recommend using [conda-forge/miniforge with mamba](https://github.com/conda-forge/miniforge#mambaforge), but any Anaconda distribution should work.

1. Install conda.
2. Create the environement with `.yml` file:
    ```bash
    conda create --name OpenSemShift -f environment.yml
    ```

Any command mentioned further in this file will assume that you have activated the Anaconda environment using `conda activate OpenSemShift`.

#### Pip setup
If the environment was not created with the ```.yml``` file, follow the following steps to setup the project with Pip.
1. Install Python 3.8. (This specific version is required.)
2. Install the necessary packages:
    ```bash
    pip3 install -r requirements.txt
    ```

### Dataset
The datasets come from [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/). To download the datasets, do as follows:

For English data:
```bash
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip
```

For German data:
```bash
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_ger.zip
```

## Usage instructions
### Basic usage
To get an overview of the features provided by these tools, use:
```bash
python g5_tools.py --help
```

### Reproduce the experiments mentioned in the report
To reproduce the experiments mentioned in the report, run the following commands:

Prepare corpus for fine-tuning mBERT on MLM objective:
```bash
python build_semeval_lm_train_test.py  --corpus_paths PathToCorpus --target_path PathToTargets --language language --lm_train_test_folder PathToSaveTrainTestFiles
```
- this outputs a ```train.txt``` and ```test.txt``` from the corpus

Fine-tuning mBERT for one time slice:
```bash
python g5_tools.py bert train --train PathToSaveTrainTestFiles/train.txt --out DirPathToTrainedModel --test PathToSaveTrainTestFiles/test.txt --epochs num --batch_size num
```

Extract target word embeddings from a filtered corpus and the fine-tuned mBERT, at one time slice: 
```bash
python g5_tools.py bert filter_dataset_and_extract --slice_label NameofCorpusTimeSlice --corpus_filepath PathToCorpusToBeQueried --pathToFineTunedModel PathToModel.bin  --wordlist_path PathToTargetWordList --keep_size NumberOfLinesToBeKeptInFilteredCorpus --lang CorpusLanguage
```
- for efficiency, this function will automatically filter only the sentences containing at least one word stems that match the word list stems. ```keep_size``` can be used to specify how many lines to keep out of the filtered corpus. 
- ```pathToFineTunedModel``` should lead to a file named ```pytorch_model.bin``` (obtained by the fine-tuning command)
- this outputs a ```NameofCorpusTimeSlice.pickle```


After extracting embeddings from two time slices, measure semantic shift across two ```.pickle``` files (embeddings files extracted from models of two time slices):
```bash
python measure_semantic_shift_merged.py --method WD or JSD -corpus_slices CorpusSliceNamesSeparatedBy ";" --results_dir_path PathToTheFolderToSaveResults --embeddings_path PathToPickleFiles 
```
- this outputs `PathToTheFolderToSaveResults/word_ranking_results_METHOD.csv`

Evaluate semantic shift results with SemEval test data:
```bash
python evaluate.py --task 'semeval' --gold_standard_path PathToSemEvalGraded.txt --results_path PathToResultsFolder --corpus_slices CorpusSliceNamesSeparatedBy ";"
```
- ```results_path``` should be the output of the previous command, eg. `PathToTheFolderToSaveResults/word_ranking_results_METHOD.csv`
- ```corpus_slices``` should be the same as above




