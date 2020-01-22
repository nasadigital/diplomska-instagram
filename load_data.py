import csv
import gensim
from itertools import accumulate
import numpy as np
import pickle
import random
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

def eval_predictions(filepath, dist_path, model_path, cur_set, eval_model):
    found = [0] * 11
    documents = DocumentIterator(filepath)
    model = gensim.models.Word2Vec.load(model_path)
    with open(dist_path, 'rb') as dist:
        for entry in documents:
            filtered_entry = list(filter(lambda x: x in model.wv.vocab, entry))
            for inst in zip(pickle.load(dist), entry):
                if inst[0] == cur_set and inst[1] in filtered_entry:
                    pos = list(filter(lambda x: x is not inst[1], filtered_entry))
                    if pos:
                        eval_model(pos, inst[1], model, found)
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
        eval_model, result_path='./results.txt', split_data=False, check=1):
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
    for i in range(check):
        results.append(eval_predictions(
            filepath, dist_path, '%s%03d-%03d.model' % (model_path, i, n), i, eval_model))
        print('Finished testing model no. %d' % i)
    print('Finished testing models.')
    with open(result_path, 'w+') as res_file:
        res_file.write(str(results))
    formated_results = format_results(results)
    print(formated_results)
    with open(result_path + 'f', 'w+') as res_file:
        res_file.write(formated_results)

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

def train_test_universal_encoder(filepath, dist_path, model_path, n,
        result_path='.results.txt', split_data=False, check=1):
    tf.compat.v1.enable_eager_execution()
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    ue_model = hub.load(module_url)

    def setup_universal_encoder(documents, save_path):
        model = gensim.models.Word2Vec(documents, size=512, window=100,
                min_count=3, workers=2, sg=1)
        print("Model initialized.")
        for word in model.wv.vocab:
            model.wv[word] = ue_model([word]).numpy()[0]
        print("Vectors calculated.")
        model.save(save_path)

    def eval_universal_encoder(sample, output, model, res):
        try:
            res[[
                _[0] for _ in model.wv.similar_by_vector(ue_model([' '.join(sample)]).numpy()[0])
            ].index(output)] += 1
        except:
            res[10] += 1

    train_test_pipeline(filepath, dist_path, model_path, n,
            setup_universal_encoder, eval_universal_encoder,
            result_path=result_path, split_data=split_data, check=check)

def get_hashtags(post_content, model):
    return [i[1:] for i in post_content.split(' ') if
            i.startswith('#') and i[1:] in model.wv.vocab]
