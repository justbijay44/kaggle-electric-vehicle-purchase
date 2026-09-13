import os
import kagglehub
import pandas as pd

COMPETITION = "playground-series-s6e9"

def download_data():
    return kagglehub.competition_download(COMPETITION)

def load_data(path=None):
    path = path or download_data()

    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))
    sample_sub = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

    return train_df, test_df, sample_sub

