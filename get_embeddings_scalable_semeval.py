import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pickle
import gc
import re
from collections import defaultdict
from tokenizers import (BertWordPieceTokenizer)
from sklearn.metrics.pairwise import cosine_similarity


def remove_mentions(text, replace_token):
    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_targets(input_path, lang, tokenizer):
    if lang == 'swedish_multi':
        input_path = 'data/semeval_data/swedish/targets.txt'
    targets_dict = {}
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f:
            target = line.strip()
            if lang=='english':
                target_no_pos = target[:-3]
                targets_dict[target_no_pos] = target
            elif lang=='swedish_multi':
                targets_dict["".join(tokenizer.tokenize(target)).replace('##', '')] = target
            else:
                targets_dict[target] = target
    return targets_dict


def add_embedding_to_list(previous, word_emb):
    embeds = [x[0] / x[1] for x in previous]
    cs = list(cosine_similarity(word_emb.reshape(1, -1), np.array(embeds))[0].tolist())
    if len(previous) < 200 and max(cs) < 0.99:
        max_idx = len(previous)
        previous.append((word_emb, 1))
    else:
        max_idx = cs.index(max(cs))
        old_embd, count = previous[max_idx]
        new_embd = old_embd + word_emb
        count = count + 1
        previous[max_idx] = (new_embd, count)
    return previous, max_idx


def tokens_to_batches(ds, tokenizer, batch_size, max_length, target_words, lang):

    batches = []
    batch = []
    batch_counter = 0

    frequencies = defaultdict(int)

    if lang == 'swedish_multi':
        target_words = list(target_words.values())
    else:
        target_words = list(target_words.keys())

    print('Dataset: ', ds)
    counter = 0
    sent_counter = 0
    sent2count = {}
    count2sent = {}
    with open(ds, 'r', encoding='utf8') as f:

        for line in f:
            counter += 1

            if counter % 50000 == 0:
                print('Num articles: ', counter)

            text = line.strip()
            contains = False

            for w in target_words:
                if w.strip() in set(text.split()):
                    frequencies[w] += text.count(w)
                    contains = True

            if contains:

                if lang=='swedish':

                    tokenized_text = tokenizer.encode(text)
                    tokens = tokenized_text.tokens
                    if len(tokens) > max_length:
                        tokens = tokens[1:max_length - 1]
                        text = " ".join(tokens).replace(" ##", "")
                        tokenized_text = tokenizer.encode(text)
                        tokens = tokenized_text.tokens
                    sent = " ".join(tokens).replace(" ##", "")
                else:
                    marked_text =  "[CLS] " + text + " [SEP]"
                    tokenized_text = tokenizer.tokenize(marked_text)
                    if len(tokenized_text) > max_length:
                        tokenized_text = tokenized_text[:max_length - 1] + ['[SEP]']
                    sent = tokenizer.convert_tokens_to_string(tokenized_text)
                count2sent[sent_counter] = sent
                sent2count[sent] = sent_counter
                batch_counter += 1
                if lang=='swedish':
                    input_sequence = tokenized_text.tokens
                    indexed_tokens = tokenized_text.ids
                else:
                    input_sequence = tokenized_text
                    indexed_tokens = tokenizer.convert_tokens_to_ids(input_sequence)

                batch.append((indexed_tokens, input_sequence))

                if batch_counter % batch_size == 0:
                    batches.append(batch)
                    batch = []

    print()
    print('Tokenization done!')
    print('len batches: ', len(batches))
    print('Word frequencies:')

    for w, freq in frequencies.items():
        print(w + ': ', str(freq))

    return batches, count2sent, sent2count


def get_token_embeddings(batches, model, batch_size):

    encoder_token_embeddings = []
    tokenized_text = []
    counter = 0

    for batch in batches:
        counter += 1
        if counter % 1000 == 0:
            print('Generating embedding for batch: ', counter)
        lens = [len(x[0]) for x in batch]
        max_len = max(lens)
        tokens_tensor = torch.zeros(batch_size, max_len, dtype=torch.long).cuda()
        segments_tensors = torch.ones(batch_size, max_len, dtype=torch.long).cuda()
        batch_idx = [x[0] for x in batch]
        batch_tokens = [x[1] for x in batch]

        for i in range(batch_size):
            length = len(batch_idx[i])
            for j in range(max_len):
                if j < length:
                    tokens_tensor[i][j] = batch_idx[i][j]

        #print("Input shape: ", tokens_tensor.shape)
        #print(tokens_tensor)

        # Predict hidden states features for each layer
        with torch.no_grad():
            model_output = model(tokens_tensor, segments_tensors)
            encoded_layers = model_output[-1][1:]  #last four layers of the encoder


        for batch_i in range(batch_size):
            encoder_token_embeddings_example = []
            tokenized_text_example = []


            # For each token in the sentence...
            for token_i in range(len(batch_tokens[batch_i])):

                # Holds 12 layers of hidden states for each token
                hidden_layers = []

                # For each of the 12 layers...
                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][batch_i][token_i]

                    hidden_layers.append(vec)

                hidden_layers = torch.sum(torch.stack(hidden_layers)[-4:], 0).reshape(1, -1).detach().cpu().numpy()

                encoder_token_embeddings_example.append(hidden_layers)
                tokenized_text_example.append(batch_tokens[batch_i][token_i])

            encoder_token_embeddings.append(encoder_token_embeddings_example)
            tokenized_text.append(tokenized_text_example)


        # Sanity check the dimensions:
        #print("Number of tokens in sequence:", len(token_embeddings))
        #print("Number of layers per token:", len(token_embeddings[0]))

    return encoder_token_embeddings, tokenized_text


def get_time_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length, lang, target_dict, concat=False):
    vocab_vectors = {}
    count2sents = {}

    for ds in datasets:

        period = ds[-5:-4]

        all_batches, count2sent, sent2count = tokens_to_batches(ds, tokenizer, batch_size, max_length, target_dict, lang)

        count2sents[period] = count2sent
        targets = set(list(target_dict.keys()))
        #print(targets)
        chunked_batches = chunks(all_batches, 1000)
        num_chunk = 0

        for batches in chunked_batches:
            num_chunk += 1
            print('Chunk ', num_chunk)

            #get list of embeddings and list of bpe tokens
            encoder_token_embeddings, tokenized_text = get_token_embeddings(batches, model, batch_size)

            splitted_tokens = []
            if not concat:
                encoder_splitted_array = np.zeros((1, 768))
            else:
                encoder_splitted_array = []
            prev_token = ""
            encoder_prev_array = np.zeros((1, 768))
            sent_tokens = []

            #go through text token by token
            for example_idx, example in enumerate(tokenized_text):
                for i, token_i in enumerate(example):
                    if token_i == "[CLS]":
                        last_start = i
                    elif token_i == "[SEP]":
                        last_finish = i
                        # print('Example: ', len(example), example)
                        if lang != 'swedish':
                            sentence = tokenizer.convert_tokens_to_string(example[last_start:last_finish + 1])
                        else:
                            sentence = " ".join(example[last_start:last_finish + 1]).replace(" ##", "")
                        # we ignore sents that span across two sequences
                        if sentence.startswith('[CLS]') and sentence.endswith('[SEP]'):
                            # print('Sentence: ', sentence)
                            sentence = sent2count[sentence]
                            # print('Count: ', sentence)
                            for sent_token, sent_idx in sent_tokens:
                                # print(sent_token, count2sent[sentence])
                                if sent_idx in vocab_vectors[sent_token][period + '_text']:
                                    vocab_vectors[sent_token][period + '_text'][sent_idx].append(sentence)
                                else:
                                    vocab_vectors[sent_token][period + '_text'][sent_idx] = [sentence]
                            sent_tokens = []

                    encoder_array = encoder_token_embeddings[example_idx][i]

                    #word is split into parts
                    if token_i.startswith('##'):

                        #add words prefix (not starting with ##) to the list
                        if prev_token:
                            splitted_tokens.append(prev_token)
                            prev_token = ""
                            if not concat:
                                encoder_splitted_array = encoder_prev_array
                            else:
                                encoder_splitted_array.append(encoder_prev_array)


                        #add word to splitted tokens array and add its embedding to splitted_array
                        splitted_tokens.append(token_i)
                        if not concat:
                            encoder_splitted_array += encoder_array
                        else:
                            encoder_splitted_array.append(encoder_array)

                    #word is not split into parts
                    else:
                        if token_i in targets:
                            #print(token_i)
                            if i == len(example) - 1 or not example[i + 1].startswith('##'):
                                if target_dict[token_i] in vocab_vectors:
                                    #print("In vocab: ", token_i + '_' + period, list(vocab_vectors.keys()))
                                    if period in vocab_vectors[target_dict[token_i]]:
                                        previous = vocab_vectors[target_dict[token_i]][period]
                                        new, new_idx = add_embedding_to_list(previous, encoder_array.squeeze())
                                        vocab_vectors[target_dict[token_i]][period] = new
                                        sent_tokens.append((target_dict[token_i], new_idx))
                                    else:
                                        vocab_vectors[target_dict[token_i]][period] = [(encoder_array.squeeze(), 1)]
                                        vocab_vectors[target_dict[token_i]][period + '_text'] = {}
                                        sent_tokens.append((target_dict[token_i], 0))
                                else:
                                    #print("Not in vocab yet: ", token_i + '_' + period, list(vocab_vectors.keys()))
                                    vocab_vectors[target_dict[token_i]] = {period: [(encoder_array.squeeze(), 1)], period + '_text': {}}
                                    sent_tokens.append((target_dict[token_i], 0))
                        #check if there are words in splitted tokens array, calculate average embedding and add the word to the vocabulary
                        if splitted_tokens:
                            if not concat:
                                encoder_sarray = encoder_splitted_array / len(splitted_tokens)
                            else:
                                encoder_sarray = np.concatenate(encoder_splitted_array, axis=1)
                            stoken_i = "".join(splitted_tokens).replace('##', '')

                            if stoken_i in targets:
                                if target_dict[stoken_i] in vocab_vectors:
                                    #print("S In vocab: ", stoken_i + '_' + period, list(vocab_vectors.keys()))
                                    if period in vocab_vectors[target_dict[stoken_i]]:
                                        previous = vocab_vectors[target_dict[stoken_i]][period]
                                        new, new_idx = add_embedding_to_list(previous, encoder_sarray.squeeze())
                                        vocab_vectors[target_dict[stoken_i]][period] = new
                                        sent_tokens.append((target_dict[stoken_i], new_idx))
                                    else:
                                        vocab_vectors[target_dict[stoken_i]][period] = [(encoder_sarray.squeeze(), 1)]
                                        vocab_vectors[target_dict[stoken_i]][period + '_text'] = {}
                                        sent_tokens.append((target_dict[stoken_i], 0))
                                else:
                                    # print("S Not in vocab yet: ", stoken_i + '_' + period, list(vocab_vectors.keys()))
                                    vocab_vectors[target_dict[stoken_i]] = {period: [(encoder_sarray.squeeze(), 1)], period + '_text': {}}
                                    sent_tokens.append((target_dict[stoken_i], 0))
                            splitted_tokens = []
                            if not concat:
                                encoder_splitted_array = np.zeros((1, 768))
                            else:
                                encoder_splitted_array = []

                        encoder_prev_array = encoder_array
                        prev_token = token_i

            del encoder_token_embeddings
            del tokenized_text
            del batches
            gc.collect()

            '''for k, v in vocab_vectors.items():
                print(k)
                input = v[0]
                encoder = v[1]
                context = v[2]
                print(len(input))
                print(len(encoder))
                print(len(context))
                print(context[0])'''

        print('Sentence embeddings generated.')

    print("Length of vocab after training: ", len(vocab_vectors.items()))

    with open(embeddings_path.split('.')[0] + '.pickle', 'wb') as handle:
        pickle.dump([vocab_vectors, count2sents], handle, protocol=pickle.HIGHEST_PROTOCOL)

    gc.collect()


if __name__ == '__main__':
    batch_size = 8
    max_length = 256

    data_folder = 'data/semeval_data/'

    langs = ['swedish', 'english', 'german', 'latin']
    concats = [False]
    fine_tuned = True

    for lang in langs:
        for concat in concats:
            if lang in ['latin', 'english']:
                datasets = [data_folder + lang + '/' + lang + '_clean_1.txt',
                            data_folder + lang + '/' + lang + '_clean_2.txt',]
            elif lang == 'swedish_multi':
                datasets = [data_folder + 'swedish/swedish_1.txt',
                            data_folder + 'swedish/swedish_2.txt', ]
            else:
                datasets = [data_folder + lang + '/' + lang + '_1.txt',
                            data_folder + lang + '/' + lang + '_2.txt', ]


            if lang == 'swedish':
                tokenizer = BertWordPieceTokenizer("data/semeval_data/swedish/vocab_swebert.txt", lowercase=True, strip_accents=False)
                if fine_tuned:
                    state_dict = torch.load("models/model_swedish/epoch_5/pytorch_model.bin")
                    model = BertModel.from_pretrained('af-ai-center/bert-base-swedish-uncased', state_dict=state_dict, output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/swedish_5_epochs_concat_scalable.pickle'
                    else:
                        embeddings_path = 'embeddings/swedish_5_epochs_scalable.pickle'
                else:
                    model = BertModel.from_pretrained('af-ai-center/bert-base-swedish-uncased', output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/swedish_pretrained_concat_scalable.pickle'
                    else:
                        embeddings_path = 'embeddings/swedish_pretrained_scalable.pickle'
            elif lang == 'german':
                tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
                if fine_tuned:
                    state_dict = torch.load("models/model_german_cased/epoch_5/pytorch_model.bin")
                    model = BertModel.from_pretrained('bert-base-german-cased', state_dict=state_dict, output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/german_5_epochs_concat_scalable.pickle'
                    else:
                        embeddings_path = 'embeddings/german_5_epochs_scalable.pickle'
                else:
                    model = BertModel.from_pretrained('bert-base-german-cased', output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/german_pretrained_concat_scalable.pickle'
                    else:
                        embeddings_path = 'embeddings/german_pretrained_scalable.pickle'

            elif lang == 'english':
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
                if fine_tuned:
                    state_dict = torch.load("models/model_english_epoch_8/epoch_5/pytorch_model.bin")
                    model = BertModel.from_pretrained('bert-base-uncased', state_dict=state_dict, output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/english_5_epochs_concat_scalable.pickle'
                    else:
                        embeddings_path = 'embeddings/english_5_epochs_scalable.pickle'
                else:
                    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/english_pretrained_concat_scalable.pickle'
                    else:
                        embeddings_path = 'embeddings/english_pretrained_scalable.pickle'

            elif lang == 'latin':
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
                if fine_tuned:
                    state_dict = torch.load("models/model_latin/epoch_5/pytorch_model.bin")
                    model = BertModel.from_pretrained('bert-base-multilingual-uncased', state_dict=state_dict, output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/latin_5_epochs_concat_scalable.pickle'
                    else:
                        embeddings_path = 'embeddings/latin_5_epochs_scalable.pickle'
                else:
                    model = BertModel.from_pretrained('bert-base-multilingual-uncased', output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/latin_pretrained_concat_scalable.pickle'
                    else:
                        embeddings_path = 'embeddings/latin_pretrained_scalable.pickle'

            elif lang == 'swedish_multi':
               tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
               if fine_tuned:
                   state_dict = torch.load("models/model_swedish_multilingual/pytorch_model.bin")
                   model = BertModel.from_pretrained('bert-base-multilingual-uncased', state_dict=state_dict, output_hidden_states=True)
                   if concat:
                       embeddings_path = 'embeddings/swedish_multi_concat_scalable.pickle'
                   else:
                       embeddings_path = 'embeddings/swedish_multi_scalable.pickle'
               else:
                   model = BertModel.from_pretrained('bert-base-multilingual-uncased', output_hidden_states=True)
                   if concat:
                       embeddings_path = 'embeddings/swedish_multi_pretrained_concat_scalable.pickle'
                   else:
                       embeddings_path = 'embeddings/swedish_multi_pretrained_scalable.pickle'

            model.cuda()
            model.eval()

            target_dict = get_targets(data_folder + lang + '/targets.txt', lang, tokenizer)


            print(target_dict)

            get_time_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length, lang, target_dict=target_dict, concat=concat)


