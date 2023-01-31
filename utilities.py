import pandas as pd


def read_data(data_path, test=False):
    """_summary_

    Args:
        data_path (_type_): data path of file to read
        test (bool, optional): whether data is test (without score columns)

    Returns:
        tuple: (sample_names, features, scores) all are numpy arrays
    """
    df = pd.read_csv(data_path, header=None)
    sample_names = df.iloc[:, 0].to_numpy()
    if not test:
        scores, features = df.iloc[:, 1].to_numpy(), df.iloc[:, 2:].to_numpy()
    else:
        scores, features = None, df.iloc[:, 1:].to_numpy()
    
    return sample_names, features, scores

    
    