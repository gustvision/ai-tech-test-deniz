import pandas as pd

if __name__ == '__main__':
    csv_file = 'data/whale-detection-challenge/whale_data/data/train.csv'
    data = pd.read_csv(csv_file)

    train = data.sample(frac=0.8, random_state=1337)
    val = data[~data.index.isin(train.index)]
    train.to_csv("data/whale-detection-challenge/whale_data/data/custom_train.csv", index=False)
    val.to_csv("data/whale-detection-challenge/whale_data/data/custom_val.csv", index=False)
