from plackett_luce import datasets as ds


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
    df.to_csv('../data/example_data/example_data.csv', index=False)


if __name__ == '__main__':
    save_dataframe()
