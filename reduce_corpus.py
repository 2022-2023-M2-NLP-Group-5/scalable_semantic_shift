import os
import argparse
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.cistem import Cistem

def reduce_corpus(corpus_filepath,
                  output_filepath,
                  target_filepath,
                  lang,
                  self_defined_words = "",
                  semeval_eng = False):
    corpus = open(corpus_filepath).readlines()
    if lang == "english":
        stemmer = SnowballStemmer("english")
    elif lang == "german":
        stemmer = Cistem()

    if self_defined_words != "": 
        target = self_defined_words.split(";")
    else:
        targets = open(target_filepath).readlines()
        if semeval_eng == True:
            target = set([tok if len(tok:= stemmer.stem(t.split("_")[0])) < 4 else tok[:4] for t in targets])
            #target = set([stemmer.stem(t.split("_")[0])[:-2] for t in targets])
        else:
            target = set([tok if len(tok:= stemmer.stem(t.split("\n")[0])) < 4 else tok[:4] for t in targets])
            #target = set([stemmer.stem(t.split("\n")[0])[:-2] for t in targets])

    print("Querying these words:", target)

    keep = count = 0
    with open(os.path.join(output_filepath), "w") as f:
        for line in corpus:
            #text = {tok if len(tok:= stemmer.stem(word)) < 4 else tok[:-1] for word in set(line.split())}
            text = {x for x in set(line.split()) for i in target if (x.startswith(i))}
            count += 1
            #if target.intersection(text) == set():
            if len(text) == 0:
                continue
            else:
                keep += 1
                f.write(line)
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
    parser.add_argument("--self_defined_words",
                        default="",
                        type=str,
                        help="each word separated by ;")
    parser.add_argument("--semeval_eng",
                        default=False,
                        type=bool,
                        help="True only if the targets are from semeval english")
    args = parser.parse_args()

    reduce_corpus(corpus_filepath = args.corpus_filepath,
                  output_filepath = args.output_filepath,
                  target_filepath = args.target_filepath,
                  lang = args.lang,
                  self_defined_words = args.self_defined_words,
                  semeval_eng = args.semeval_eng)

