from azureml.core import Dataset, Run
from load_data import prep_train_test_bert
import random

random.seed(4)

run = Run.get_context()
workspace = run.experiment.workspace

dataset = Dataset.get_by_name(workspace=workspace, name='dataset')
dataset.download(target_path='.', overwrite=False)

dist = Dataset.get_by_name(workspace=workspace, name='dataset_dist')
dist.download(target_path='.', overwrite=False)

prep_train_test_bert('./media.csv', './dist.dat', './models/1024dRoBertAModel',
        10, result_path='./result1024dRoBertA.txt', check=1,
        pretrained_weights='roberta-base')
