import numpy as np
from constants import classes
from data import Dataset
from cnn import CNN
from tqdm import tqdm

### PARAMS
BATCH_SIZE = 32
NUM_CLASSES = 10
LEARNING_RATE = 1e-2
NUM_EPOCHS = 100
PRINT_INTERVAL = 2000
SMOOTHING_FACTOR = 0.9

dataset = Dataset("./data/cifar-10-batches-py", batch_size=BATCH_SIZE, sample_interval=1)
model = CNN(NUM_CLASSES=NUM_CLASSES, learning_rate=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    ## TRAINING
    for batch_idx in tqdm(range(dataset.num_train_batches)):
        X, y = dataset.train_batch(batch_idx)
        batch_size = X.shape[0]
        one_hot = np.zeros((y.shape[0], NUM_CLASSES))
        one_hot[np.arange(y.shape[0]), y] = 1
        prob = model.forward(X)
        loss = np.sum ( -1 * np.log(prob) * one_hot ) / batch_size
        del_prob = -1 * one_hot / prob
        del_prob /= batch_size
        model.backward(del_prob)
        model.update_weights()

    ## TRAINING ACC
    print("FINDING TRAINING ACC ... ")
    train_correct = 0
    for batch_idx in tqdm(range(dataset.num_train_batches)):
        X, y = dataset.train_batch(batch_idx)
        batch_size = X.shape[0]
        prob = model.forward(X)
        train_correct += np.sum(np.argmax(prob, axis=1) == y)
            
    ## TESTING ACC
    print("TESTING ... ")
    test_correct = 0
    for batch_idx in tqdm(range(dataset.num_test_batches)):
        X, y = dataset.test_batch(batch_idx)
        batch_size = X.shape[0]
        prob = model.forward(X)
        test_correct += np.sum(np.argmax(prob, axis=1) == y)
    print(f"Train Accuracy: {train_correct/dataset.num_train_samples*100:.2f}% ||| Validation Accuracy: {test_correct/dataset.num_test_samples*100:.2f} %")
    