import pandas as pd
import matplotlib.pyplot as plt


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


def plot_loss(train_loss, val_loss):
    
    """_summary_
        train_loss : list of train MSE values
        val_loss : list of val MSE values
    """
    
    plt.plot(train_loss, label="Train MSE")
    plt.plot(val_loss, label="Val MSE")
    plt.grid(True)
    plt.xlabel("Number of Iterations")
    plt.ylabel("MSE Loss")
    plt.title("Loss curves")
    plt.legend()
    plt.show()
    
    
    