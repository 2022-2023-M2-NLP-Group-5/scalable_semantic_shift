import os
import argparse

def reduce_corpus(corpus_filepath,
                  output_filepath,
                  target_filepath,
                  self_defined_words = "",
                  semeval_eng = True):
    corpus = open(corpus_filepath).readlines()
    
    if self_defined_words != "": 
        target = self_defined_words.split(";")
    else:
        targets = open(target_filepath).readlines()
        if semeval_eng == True:
            target = [t.split("_")[0] for t in targets]
        else:
            target = [t.split("\n")[0] for t in targets]
    target = set(target)
    print("Querying these words:", target)
    keep = count = 0
    with open(os.path.join(output_filepath), "w") as f:
        for line in corpus:
            count += 1
            if target.intersection(set(line.split())) == set():
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
    parser.add_argument("--self_defined_words",
                        default="",
                        type=str,
                        help="each word separated by ;")
    parser.add_argument("--semeval_eng",
                        default=False,
                        type=bool,
                        help="True only if the targets are from semeval english")
    args = parser.parse_args()

    reduce_corpus(args.corpus_filepath, args.output_filepath, args.target_filepath, args.self_defined_words,args.semeval_eng)

