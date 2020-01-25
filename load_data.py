import csv
import gensim
from itertools import accumulate
import numpy as np
import pickle
import random
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import tensorflow_hub as hub
import time

class UserLink(object):
    def __init__(self, row):
        self.to = int(row[1])
        self.likes = int(row[2])
        self.comments = [
           int(timestamp) for timestamp in row[4].split(',')
        ] if row[4] else []

    def __repr__(self):
        return '%d %d %d' % (self.to, self.likes, len(self.comments))

class Media(object):
    def __init__(self, row, hashtags):
        self.id = int(row[0])
        self.timestamp = int(row[2])
        if len(row) == 7:
            self.likes = int(row[4])
            self.comments = int(row[5])
            for hashtag in row[3].split(','):
                if hashtag not in hashtags:
                    hashtags[hashtag] = len(hashtags) - 1
            self.tags = [hashtags[hashtag] for hashtag in row[3].split(',')]
        else:
            self.likes = int(row[3])
            self.comments = int(row[4])
            self.tags = []

    def __repr__(self):
        return '%d %d %d %d %d' % (self.id, self.timestamp, self.likes,
                                   self.comments, len(self.tags)) 

def load_users(filepath):
    graph = {}
    with open(filepath) as users_file:
        csv_read = csv.reader(users_file, delimiter=';')
        next(csv_read, None)
        for row in csv_read:
            if int(row[0]) not in graph:
                graph[int(row[0])] = []
            graph[int(row[0])].append(UserLink(row))
    return graph

def load_media(filepath):
    media = {}
    hashtags = {}
    with open(filepath, encoding='ISO-8859-1') as media_file:
        csv_read = csv.reader(media_file, delimiter=';')
        next(csv_read, None)
        for row in csv_read:
            if int(row[1]) not in media:
                media[int(row[1])] = []
            media[int(row[1])].append(Media(row, hashtags))
    return media, hashtags

class DocumentIterator:
    def __init__(self, filepath, dist_path='', cur_set=-1):
        self.filepath = filepath
        self.dist_path = dist_path
        self.cur_set = cur_set

    def __iter__(self):
        with open(self.filepath, encoding='ISO-8859-1') as media_file:
            csv_read = csv.reader(media_file, delimiter=';')
            next(csv_read, None)
            dist = None
            if (self.dist_path):
                dist = open(self.dist_path, 'rb')
            for row in csv_read:
                if len(row) == 7:
                    if dist:
                        filtered = list(map(lambda x: x[1], filter(
                            lambda x: x[0] != self.cur_set,
                            zip(pickle.load(dist), row[3].split(',')))))
                        if not filtered:
                            continue
                        yield filtered
                    else:
                        yield row[3].split(',')

def generate_embeddings(filepath, save_path, train_model, dist_path='',
                        cur_set=-1):
    start_time = time.time()
    documents = list(DocumentIterator(filepath, dist_path=dist_path,
                                      cur_set=cur_set))
    print("Docs loaded.")
    train_model(documents, save_path)
    print("Training took: {0}".format(time.time() - start_time))

def split_nway(filepath, save_path, n=10):
    with open(save_path, 'wb+') as split_file:
        documents = DocumentIterator(filepath)
        for entry in documents:
            pickle.dump([random.randint(0, n - 1) for _ in entry], split_file)

def eval_predictions(filepath, dist_path, model_path, cur_set, load_model,
        eval_model):
    found = [0] * 11
    documents = DocumentIterator(filepath)
    last_checkpoint = time.time()
    vocab, model = load_model(model_path)
    batch_in = []
    batch_out = []
    with open(dist_path, 'rb') as dist:
        for idx, entry in enumerate(documents):
            if idx % 50000 == 0:
                print("{0} examples classified: {1} s".format(
                    idx, time.time() - last_checkpoint))
            filtered_entry = list(filter(lambda x: x in vocab, entry))
            for inst in zip(pickle.load(dist), entry):
                if inst[0] == cur_set and inst[1] in filtered_entry:
                    pos = list(filter(lambda x: x is not inst[1], filtered_entry))
                    if pos:
                        batch_in.append(pos)
                        batch_out.append(inst[1])
                        if idx % 500 == 0:
                            eval_model(batch_in, batch_out, model, found)
                            batch_in.clear()
                            batch_out.clear()
    eval_model(batch_in, batch_out, model, found)
    print("Done evaluating: {0} s".format(time.time() - last_checkpoint))
    return found

def format_results(results):
    lines = []
    cum_res = [0] * 11
    for res in results:
        lines.append(str(res))
        lines.append(str(list(accumulate(res))))
        lines.append(
                str(['%.4f' % (i / sum(res)) for i in list(accumulate(res))]))
        for pair in zip(range(11), list(accumulate(res))):
            cum_res[pair[0]] += pair[1] / sum(res)
    lines.append(str(['%.4f' % (i / len(results)) for i in cum_res]))
    return '\n'.join(lines)

def train_test_pipeline(filepath, dist_path, model_path, n, train_model,
        load_model, eval_model, result_path='./results.txt', split_data=False,
        check=1):
    if split_data:
        print('Generating split...')
        split_nway(filepath, dist_path, n)
    print('Generated split.\nTraining models...')
    for i in range(check):
        generate_embeddings(filepath, '%s%03d-%03d.model' % (model_path, i, n),
                train_model, dist_path=dist_path, cur_set=i)
        print('Finished training model no. %d' % i)
    print('Finished training all models.\nEvaluating models...')
    results = []
    fromated_results = ''
    for i in range(check):
        results.append(eval_predictions(
            filepath, dist_path, '%s%03d-%03d.model' % (model_path, i, n), i,
            load_model, eval_model))
        with open(result_path + 'f', 'w+') as res_file:
            formated_results = format_results(results)
            res_file.write(formated_results)
        print('Finished testing model no. %d' % i)
    print('Finished testing models.')
    with open(result_path, 'w+') as res_file:
        res_file.write(str(results))
    print(formated_results)

class EpochLogger(gensim.models.callbacks.CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

def train_test_hashtag2vec(filepath, dist_path, model_path, n, dim=64, epochs=10,
        result_path='./results.txt', split_data=False, check=1, sg=1):

    def train_hashtag2vec(documents, save_path):
        epoch_logger = EpochLogger()
        model = gensim.models.Word2Vec(documents, size=dim, window=100,
                min_count=3, workers=2, sg=sg)
        print("Model initialized.")
        model.train(documents, total_examples=len(documents),
                    epochs=epochs, callbacks=[epoch_logger])
        print('Finished training!\nSaving...')
        model.save(save_path)

    def load_hashtag2vec(model_path):
        model = gensim.models.Word2Vec.load(model_path)
        return model

    def eval_hashtag2vec(sample, output, model, res): 
        try:
            res[[
                _[0] for _ in model.wv.most_similar(positive=sample)
            ].index(output)] += 1
        except:
            res[10] += 1

    train_test_pipeline(filepath, dist_path, model_path, n, train_hashtag2vec,
                        eval_hashtag2vec, result_path=result_path,
                        split_data=split_data, check=check)

def build_vocab(docs):
    occ = {}
    for line in docs:
        for word in line:
            occ[word] = occ.get(word, 0) + 1
    vocab = []
    for kv in occ.items():
        if kv[1] > 2:
            vocab.append(kv[0])
    return vocab

def save_word2vec_format(path, vector_dict):
    with open(path, 'w') as writer:
        writer.write("{0} {1}\n".format(len(vector_dict),
                                        len(next(iter(vector_dict.values())))))
        for kv in vector_dict.items():
            writer.write("{0} {1}\n".format(kv[0], ' '.join([str(val) for val in kv[1]])))
    print("Saved {0}".format(path))

def train_test_universal_encoder_balltree(filepath, dist_path, model_path, n,
        result_path='.results.txt', split_data=False, check=1):
    tf.compat.v1.enable_eager_execution()
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    ue_model = hub.load(module_url)
    
    def setup_universal_encoder(documents, save_path):
        last_checkpoint = time.time()
        vocabulary = build_vocab(documents)
        print("Vocabulary built: {0} s".format(time.time() - last_checkpoint))
        print("Total words: {0}".format(len(vocabulary)))
        last_checkpoint = time.time()
        calculated = ue_model(vocabulary).numpy()
        vocab_vectors = {}
        for idx, word in enumerate(vocabulary):
            vocab_vectors[word] = calculated[idx]
            if idx % 10000 == 0:
                print("{0} words done: {1}\nLast word: {2}".format(
                    idx, time.time() - last_checkpoint, word))
        print("Vectors calculated: {0} s".format(time.time() - last_checkpoint))
        last_checkpoint = time.time()
        with open(save_path, 'wb+') as model_file:
            pickle.dump(vocabulary, model_file)
            pickle.dump(NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
                    .fit([vocab_vectors[word] for word in vocabulary]), model_file)

        print("Model saved: {0} s".format(time.time() - last_checkpoint))

    def load_universal_encoder(model_path):
        with open(model_path, 'rb') as model_file:
            n_vocab = pickle.load(model_file)
            n_model = pickle.load(model_file)
        return (set(n_vocab),
                ({val : idx for idx, val in enumerate(n_vocab)}, n_model))

    def eval_universal_encoder(sample, output, model, res):
        try:
            res[list(model[1].kneighbors(
                    ue_model([' '.join(sample)]).numpy()
                )[1][0]).index(model[0].get(output, -1))] += 1
        except:
            res[10] += 1

    train_test_pipeline(filepath, dist_path, model_path, n,
            setup_universal_encoder, load_universal_encoder,
            eval_universal_encoder, result_path=result_path,
            split_data=split_data, check=check)

def train_test_universal_encoder(filepath, dist_path, model_path, n,
        result_path='.results.txt', split_data=False, check=1):
    tf.compat.v1.enable_eager_execution()
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    ue_model = hub.load(module_url)

    def setup_universal_encoder(documents, save_path):
        last_checkpoint = time.time()
        vocabulary = build_vocab(documents)
        print("Vocabulary built: {0} s".format(time.time() - last_checkpoint))
        print("Total words: {0}".format(len(vocabulary)))
        last_checkpoint = time.time()
        calculated = ue_model(vocabulary).numpy()
        vocab_vectors = {}
        for idx, word in enumerate(vocabulary):
            vocab_vectors[word] = calculated[idx]
            if idx % 10000 == 0:
                print("{0} words done: {1}\nLast word: {2}".format(
                    idx, time.time() - last_checkpoint, word))
        print("Vectors calculated: {0} s".format(time.time() - last_checkpoint))
        last_checkpoint = time.time()
        save_word2vec_format(save_path, vocab_vectors)
        print("Model saved: {0} s".format(time.time() - last_checkpoint))

    def load_universal_encoder(model_path):
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
        return (set(model.wv.vocab.keys()), model)

    def eval_universal_encoder(sample, output, model, res):
        samples = ue_model([' '.join(s[:6]) for s in sample]).numpy()
        for i in range(len(output)):
            try:
                res[[
                    _[0] for _ in model.wv.similar_by_vector(
                        samples[i]
                    )].index(output[i])] += 1
            except:
                res[10] += 1

    train_test_pipeline(filepath, dist_path, model_path, n,
            setup_universal_encoder, load_universal_encoder,
            eval_universal_encoder, result_path=result_path,
            split_data=split_data, check=check)

def get_hashtags(post_content, model):
    return [i[1:] for i in post_content.split(' ') if
            i.startswith('#') and i[1:] in model.wv.vocab]
