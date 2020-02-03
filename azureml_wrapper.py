from azureml.core import Dataset, Run
from load_data import train_test_bert, prep_train_test_bert
import random

random.seed(4)

run = Run.get_context()
workspace = run.experiment.workspace

dataset = Dataset.get_by_name(workspace=workspace, name='dataset')
dataset.download(target_path='.', overwrite=False)

dist = Dataset.get_by_name(workspace=workspace, name='dataset_dist')
dist.download(target_path='.', overwrite=False)

def bert_precompute():
    prep_train_test_bert('./media.csv', './dist.dat', './models/1024dRoBertAModel',
            10, result_path='./result1024dRoBertA.txt', check=1,
            pretrained_weights='roberta-base')

def train_mlp():
    os.makedirs(os.path.dirname('./outputs/'), exist_ok=True)
    precalced = Dataset.get_by_name(workspace, name='distilbert-base-uncased_pack')
    precalced.download(target_path='./outputs/', overwrite=False)
    train_test_bert('./media.csv', './dist.dat', './models/768dBertModel',
                    10, result_path='./result768dBert.txt', check=1,
                    pretrained_weights='distilbert-base-uncased')


train_mlp()
