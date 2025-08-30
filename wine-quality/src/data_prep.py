import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import read_yaml, ensure_dir

def main():
    params = read_yaml('params.yaml')
    raw_csv = params['data']['raw_csv']
    test_size = params['data']['test_size']
    random_state = params['data']['random_state']

    df = pd.read_csv(raw_csv)
    df = df.dropna()

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[params['data']['target']]
    )

    ensure_dir('data/processed')
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print('Saved train.csv and test.csv')

if __name__ == '__main__':
    main()