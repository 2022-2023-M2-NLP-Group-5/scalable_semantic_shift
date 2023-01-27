import os
import argparse
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.cistem import Cistem


def reduce_corpus(corpus_filepath,
                  output_filepath,
                  lang,
                  target_filepath="",
                  self_defined_words="",
                  stem_self_defined=False,
                  semeval_eng=False,
                  keep_size=80000):
    corpus = open(corpus_filepath).readlines()
    if lang == "english":
        stemmer = SnowballStemmer("english")
    elif lang == "german":
        stemmer = Cistem()

    if self_defined_words != "":
        targets = self_defined_words.split(";")
    else:
        targets = open(target_filepath).readlines()

    if (self_defined_words != "" and stem_self_defined == True) or self_defined_words == "":
        if semeval_eng == True:
            target = set([tok if len(tok := stemmer.stem(t.split("_")[0])) < 4 else tok[:4] for t in targets])
            # target = set([stemmer.stem(t.split("_")[0])[:-2] for t in targets])
        else:
            target = set([tok if len(tok := stemmer.stem(t.split("\n")[0])) < 4 else tok[:4] for t in targets])
            # target = set([stemmer.stem(t.split("\n")[0])[:-2] for t in targets])
    else:
        target = [w.lower() for w in targets]

    print("Querying these words:", target)
    count = len(corpus)
    keep = 0
    all_lines = []
    with open(os.path.join(output_filepath), "w") as f:
        for line in corpus:
            # text = {tok if len(tok:= stemmer.stem(word)) < 4 else tok[:-1] for word in set(line.split())}
            text = {x for x in set(line.split()) for i in target if (x.lower().startswith(i))}
            if len(text) == 0:
                continue
            elif keep < keep_size:
                keep += 1
                f.write(line)
            else:
                break
    print("Number of lines in reduced corpus:", keep)
    print("Number of lines in original corpus:", count)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_filepath",
                        default='data/semeval2020_ulscd_eng/corpus1/token/ccoha1.txt',
                        type=str,
                        help="")
    parser.add_argument("--output_filepath",
                        default="data/c1/english_reduced_corpus.txt",
                        type=str,
                        help="")
    parser.add_argument("--target_filepath",
                        default="data/semeval2020_ulscd_eng/targets.txt",
                        type=str,
                        help="")
    parser.add_argument("--lang",
                        default="english",
                        type=str,
                        help="english / german")
    parser.add_argument("--stem_self_defined",
                        default= False,
                        type=bool,
                        help="whether to filter with the stem the self defined word")
    parser.add_argument("--self_defined_words",
                        default="",
                        type=str,
                        help="each word separated by ;")
    parser.add_argument("--semeval_eng",
                        default=False,
                        type=bool,
                        help="True only if the targets are from semeval english")
    parser.add_argument("--keep_size",
                        default=80000,
                        type=int,
                        help="max lines to keep in the reduced corpus")
    args = parser.parse_args()

    reduce_corpus(corpus_filepath = args.corpus_filepath,
                  output_filepath = args.output_filepath,
                  target_filepath = args.target_filepath,
                  lang = args.lang,
                  self_defined_words = args.self_defined_words,
                  stem_self_defined = args.stem_self_defined,
                  semeval_eng = args.semeval_eng,
                  keep_size = args.keep_size)