import sys
import os

# Add the parent directory (the root of the project) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plackett_luce import datasets as ds
import pandas as pd


def generate_dataframe(num_samples, num_items_range=(8, 14)):
    X_, rankigns = ds.generate_data_varaible_items(num_samples=num_samples, num_items_range=num_items_range)

    df = pd.DataFrame()
    df_lt = []
    for i, Xi in enumerate(X_):
        dfi = pd.DataFrame(Xi, columns=['x'+str(i+1) for i in range(len(Xi[0]))])
        dfi['ID'] = i
        dfi['rank'] = rankigns[i]
        df_lt.append(dfi)
    return pd.concat(df_lt)



def save_dataframe():
    df = generate_dataframe(10000)
    # df.to_csv('./example_data.csv', index=False)
    df.to_csv('./data/example_data/example_data.csv', index=False)


if __name__ == '__main__':
    print(os.getcwd())
    os.makedirs("data/example_data") 
    save_dataframe()
