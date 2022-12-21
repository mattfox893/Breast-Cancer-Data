import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from scipy import stats
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier


'''
PLOTS
'''
# Creates an empty, default plot
def create_plot(title, xlabel, ylabel):
    plt.figure(figsize=(40, 10))
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)  
    plt.ylabel(ylabel, fontsize=15)

# Creates an empty, default subplot
def create_subplot(size, title, xlabel, ylabel):
    figure, ax = plt.subplots(*(1, size), figsize=(40, 10))
    return figure, ax

# Plots a Histogram of given data
def plot_histogram(df, column, title='', xlabel='', ylabel=''):
    figure, ax = create_subplot(3, title, xlabel, ylabel)

    total_freq, malignant_freq, benign_freq = 0, 0, 0
    # Combined [0]
    total_freq, _, _ = ax[0].hist(df[column], density=True, alpha=0.5, label='Combined', color='steelblue')
    # Malignant [1]
    malignant_freq, _, _ = ax[1].hist(df[df['diagnosis'] == 1][column], density=True, alpha=0.5, label='Malignant', color='salmon')
    # Benign [2]
    benign_freq, _, _ = ax[2].hist(df[df['diagnosis'] == 0][column], density=True, alpha=0.5, label='Benign', color='forestgreen')

    # Size plots
    y_upper_bound = np.max(np.concatenate((total_freq, benign_freq, malignant_freq), axis=None))
    for i in range(3):
        ax[i].set_ylim([0, y_upper_bound])
        ax[i].legend()

    # Create plot
    figure.suptitle(title, fontsize=20)
    figure.text(0.5, 0.04, xlabel, ha='center', fontsize=16)
    figure.text(0.04, 0.5, ylabel, va='center', rotation='vertical', fontsize=16)
    plt.show()

# Plots a Bar Plot of given data
def plot_bar(df, column, labels, title='', xlabel='', ylabel=''):
    create_plot(title, xlabel, ylabel)

    plt.bar(labels, list(df.groupby(column).size()), alpha=0.5, color='steelblue')

    plt.show()

# Plots a Box Plot of given data
def plot_box(data, title, xlabel, ylabel):
    create_plot(title, xlabel, ylabel)

    plt.boxplot(data)

    plt.show()

'''
BAYES
'''
# Fits a Normal Distribution to data
def fit_normal_distribution(df, column, linspace, ax, ax_number):
    normal = stats.norm.pdf(linspace, *stats.norm.fit(df[column]))  # Create normal distribution
    ax[ax_number].plot(linspace, normal, label='Normal')    # Plot distribution

# Fits a Gamma Distribution to data
def fit_gamma_distribution(df, column, linspace, ax, ax_number):
    gamma = stats.gamma.pdf(linspace, *stats.gamma.fit(df[column])) # Create gamma distribution
    ax[ax_number].plot(linspace, gamma, label='Gamma')  # Plot distribution

# Fits a Beta Distribution to data
def fit_beta_distribution(df, column, linspace, ax, ax_number):
    beta = stats.beta.pdf(linspace, *stats.beta.fit(df[column]))    # Create beta distribution
    ax[ax_number].plot(linspace, beta, label='Beta')    # Plot distribution

# Plots a Histogram of given data along with a Normal, Gamma, and Beta distribution
def plot_histogram_distribution(df, column, title='', xlabel='', ylabel=''):
    figure, ax = create_subplot(3, title, xlabel, ylabel)

    total_freq, malignant_freq, benign_freq = 0, 0, 0
    linspace = np.linspace(df[column].min(), df[column].max())

    # Combined [0]
    total_freq, _, _ = ax[0].hist(df[column], density=True, alpha=0.5, label='Combined', color='steelblue')
    fit_normal_distribution(df, column, linspace, ax, 0)
    fit_gamma_distribution(df, column, linspace, ax, 0)
    fit_beta_distribution(df, column, linspace, ax, 0)
    # Malignant [1]
    malignant_freq, _, _ = ax[1].hist(df[df['diagnosis'] == 1][column], density=True, alpha=0.5, label='Malignant', color='salmon')
    fit_normal_distribution(df[df['diagnosis'] == 1], column, linspace, ax, 1)
    fit_gamma_distribution(df[df['diagnosis'] == 1], column, linspace, ax, 1)
    fit_beta_distribution(df[df['diagnosis'] == 1], column, linspace, ax, 1)
    # Benign [2]
    benign_freq, _, _ = ax[2].hist(df[df['diagnosis'] == 0][column], density=True, alpha=0.5, label='Benign', color='forestgreen')
    fit_normal_distribution(df[df['diagnosis'] == 0], column, linspace, ax, 2)
    fit_gamma_distribution(df[df['diagnosis'] == 0], column, linspace, ax, 2)
    fit_beta_distribution(df[df['diagnosis'] == 0], column, linspace, ax, 2)

    # Size plots
    y_upper_bound = np.max(np.concatenate((total_freq, benign_freq, malignant_freq), axis=None))
    for i in range(3):
        ax[i].set_ylim([0, y_upper_bound])
        ax[i].legend()

    figure.suptitle(title, fontsize=20)
    figure.text(0.5, 0.04, xlabel, ha='center', fontsize=16)
    figure.text(0.04, 0.5, ylabel, va='center', rotation='vertical', fontsize=16)
    plt.show()

'''
SVM
'''
# Generates accuracy of running the SVM model
def run_SVM(df, kernel, number_iterations=10000):
    accuracy = np.zeros(shape=(number_iterations))

    for i in range(number_iterations):
        # Randomly split data into 85% training, 15% testing
        # Train on df columns 1...n as column 0 is the labels
        training_data, testing_data, training_label, testing_label = train_test_split(df.to_numpy()[:, 1:], df.to_numpy()[:, 0], test_size=0.15)

        # Train SVM
        classifier = svm.SVC(**kernel)
        classifier.fit(training_data, training_label)

        # Test SVM
        accuracy[i] = accuracy_score(testing_label, classifier.predict(testing_data))
    
    return accuracy, np.mean(accuracy)

'''
SINGLE LAYER PERCEPTRON
'''

def perceptron(X_train, X_test, Y_train):
    clf = Perceptron(random_state=1, max_iter=1).fit(X_train, Y_train)
    prediction_test = clf.predict(X_test)
    prediction_train = clf.predict(X_train)
    return prediction_train, prediction_test

def printPerceptron(Y_train, Y_test, prediction_train, prediction_test):
    print("Train Classification Report:")
    print(classification_report(Y_train, prediction_train))
    print("Test Classification Report:")
    print(classification_report(Y_test, prediction_test))

# Generates accuracy of running the Perceptron model
def run_Perceptron(df, number_iterations=10000):
    accuracy = np.zeros(shape=(number_iterations))

    for i in range(number_iterations):
        # Randomly split data into 85% training, 15% testing
        # Train on df columns 1...n as column 0 is the labels
        training_data, testing_data, training_label, testing_label = train_test_split(df.to_numpy()[:, 1:], df.to_numpy()[:, 0], test_size=0.15)

        # Train Perceptron
        classifier = Perceptron().fit(training_data, training_label)

        # Test Perceptron
        accuracy[i] = accuracy_score(testing_label, classifier.predict(testing_data))
    
    return accuracy, np.mean(accuracy)

'''
MLP
'''

def MLPClassify(X_train, X_test, Y_train, iterations):
    clf = MLPClassifier(random_state=1, max_iter=iterations).fit(X_train, Y_train)
    prediction_test = clf.predict(X_test)
    prediction_train = clf.predict(X_train)

    return prediction_train, prediction_test

def printMLP(Y_train, Y_test, prediction_train, prediction_test):
    print("Train Classification Report:")
    print(classification_report(Y_train, prediction_train))
    print("Test Classification Report:")
    print(classification_report(Y_test, prediction_test))


def get_score_data(X_train, Y_train, X_test, Y_test, max_iter, activation='relu'):
    training_scores = []
    test_scores = []
    iterations = []

    for i in range(1, max_iter):
        clf = MLPClassifier(random_state=1, max_iter=i, activation=activation).fit(X_train, Y_train)
        training_scores.append(clf.score(X_train, Y_train))
        test_scores.append(clf.score(X_test, Y_test))
        iterations.append(i)
    
    return iterations, training_scores, test_scores


def create_score_plots(iterations, training_scores, test_scores):
    fig, ax = plt.subplots(1, 2, figsize=(30,10))

    ax[0].plot(iterations, training_scores)
    ax[0].set_title('Training Accuracy over Iterations')
    ax[0].set_xlabel('Iterations')
    ax[0].set_ylabel('Training Accuracy')
    ax[1].plot(iterations, test_scores)
    ax[1].set_title('Test Accuracy over Iterations')
    ax[1].set_xlabel('Iterations')
    ax[1].set_ylabel('Test Accuracy')

    plt.show()


# X_train = torch.tensor(X_train.drop('diagnosis', axis=1).values.astype(np.float32))
# print(X_train)
# Y_train = torch.tensor(Y_train.values.astype(np.float32))
# X_test = torch.tensor(X_test.drop('diagnosis', axis=1).values.astype(np.float32))
# Y_test = torch.tensor(Y_test.values.astype(np.float32))
# test_data = torch.utils.data.TensorDataset(X_test, Y_test)
# train_data = torch.utils.data.TensorDataset(X_train, Y_train)
# test_dataloader = DataLoader(test_data, batch_size=64)
# train_dataloader = DataLoader(train_data, batch_size=64)

# epochs = 5
# lossfn = nn.CrossEntropyLoss()

# run_nn(epochs, train_dataloader, test_dataloader, lossfn)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)


def train_nn(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_nn(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def run_nn(epochs, train_dataloader, test_dataloader, loss_fn):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_nn(train_dataloader, model, loss_fn, optimizer)
        test_nn(test_dataloader, model, loss_fn)
        print("Done!")
        

'''
KNN
'''
def merge(list1, list2):
    ret = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return ret


#returns best K value within given K range
def find_best_K_val(range_Ks, list_classes, list_examples):
    highest_acc = 0.0
    best_K_val = 0

    for i in range(1, range_Ks + 1):
        curr_best_accuracy = 0
    
        for j in range(0,10):
            X_train, X_test, Y_train, Y_test = train_test_split(list_examples, list_classes)
            knn_classifier = KNeighborsClassifier(n_neighbors = i)
            knn_classifier.fit(X_train, Y_train)
            classifier_accuracy = accuracy_score(Y_test, knn_classifier.predict(X_test))
            curr_best_accuracy = curr_best_accuracy + classifier_accuracy

        #then calculate their average prediction rate over the 10 models where n_neighbors = i. 
        #Store the highest prediciton rate
        curr_best_accuracy = curr_best_accuracy/10.0
        if curr_best_accuracy > highest_acc :
            highest_acc = curr_best_accuracy
            best_K_val = i

    #print("Best K: %d, with accuracy: %f over %d possible K-values" %(best_K_val, highest_acc, range_Ks))
    return best_K_val, highest_acc


def mean_dict_of_lists(d):
    ks = []
    avgs = []
    for i,j in d.items():
        sum = 0.0
        for num in j:
            sum = sum + num
        ks.append(i)
        avgs.append(sum/len(j))
    return ks, avgs

def plot_bar_graph(x_vals, y_vals, title, x_label, y_label):
    plt.bar(x_vals, y_vals, width=0.5)
    plt.title(title)
    plt.xticks(range(1,21))
    plt.xlabel(x_label)
    plt.ylabel(y_label)    
    plt.show()