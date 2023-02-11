import pandas as pd
import matplotlib.pyplot as plt


def read_data(data_path, test=False, scores_last=False):
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
        if not scores_last:
            scores, features = df.iloc[:, 1].to_numpy(), df.iloc[:, 2:].to_numpy()
        else:
            ### For generalization analysis datasets
            scores, features = df.iloc[:, -1].to_numpy(), df.iloc[:, :-1].to_numpy()
            sample_names = None
    else:
        scores, features = None, df.iloc[:, 1:].to_numpy()
    
    return sample_names, features, scores


def plot_loss(train_loss, val_loss, class_loss=None, shift=True, label="MSE"):
    
    """_summary_
        train_loss : list of train MSE values
        val_loss : list of val MSE values
    """
    # plt.figure(figsize=(12,8))
    if shift:
        val_loss = [loss + 0.5 for loss in val_loss]
    plt.plot(train_loss, label=f"Train {label}")
    plt.plot(val_loss, label=f"Val {label}")
    if class_loss is not None:
        for key in class_loss.keys():
            plt.plot(class_loss[key], label=key)
    num_points = len(train_loss)
    step_size = (num_points // 20)
    if step_size == 0:
        step_size = num_points
    plt.scatter(list(range(0,num_points,step_size)) + [num_points-1] , train_loss[0:num_points:step_size] + [train_loss[-1]], s = 16, color = "red", marker="X")
    plt.scatter(list(range(0,num_points,step_size)) + [num_points-1] , val_loss[0:num_points:step_size] + [val_loss[-1]], s = 16, color = "red", marker="X")
    plt.grid(True)
    plt.xlabel("Number of Iterations")
    plt.ylabel(f"{label} Loss")
    plt.title("Loss curves")
    plt.legend()
    plt.show()
    
    
    