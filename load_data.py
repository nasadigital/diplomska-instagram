import csv
import gensim
import time
import pickle
import random

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

def generate_embeddings(filepath, save_path, dimension=64,
                        min_app=3, threads=1, no_epochs=10, skipgram=1,
                        dist_path='', cur_set=-1):
    start_time = time.time()
    documents = list(DocumentIterator(filepath, dist_path=dist_path,
                                      cur_set=cur_set))
    print('Loaded documents!')
    print(time.time() - start_time)
    model = gensim.models.Word2Vec(documents, size=dimension, window=100,
            min_count=min_app, workers=threads, sg=skipgram)
    print('Start training...')
    print(time.time() - start_time)
    model.train(documents, total_examples=len(documents),
                epochs=no_epochs)
    print('Finished training!\nSaving...')
    print(time.time() - start_time)
    model.save(save_path)

def split_nway(filepath, save_path, n=10):
    with open(save_path, 'wb+') as split_file:
        documents = DocumentIterator(filepath)
        for entry in documents:
            pickle.dump([random.randint(0, n - 1) for _ in entry], split_file)

def eval_predictions(filepath, dist_path, model_path, cur_set):
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
                        try:
                            found[[
                                _[0] for _ in model.wv.most_similar(positive=pos)
                            ].index(inst[1])] += 1
                        except:
                            found[10] += 1
    return found
