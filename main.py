import operator

import torch

import ssl

from torch.utils.data import WeightedRandomSampler, RandomSampler

from KEELDataset import KEELDataset
from neural_network import Net

ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim
import torch.nn as nn


def print_summary(net, testloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def print_detailed_summary(net, testloader, classes):
    # prepare to count predictions for each class
    accuracy_matrix = {classname: [0, 0, 0, 0] for classname in classes}

    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for sample_label, prediction in zip(labels, predictions):
                for label_index, label_value in enumerate(classes):
                    if label_index == int(prediction.item()):
                        if int(sample_label.item()) == label_index:
                            accuracy_matrix[classes[label_index]][0] += 1
                        else:
                            accuracy_matrix[classes[label_index]][1] += 1
                    else:
                        if int(sample_label.item()) != label_index:
                            accuracy_matrix[classes[label_index]][3] += 1
                        else:
                            accuracy_matrix[classes[label_index]][2] += 1
                    total_pred[classes[label_index]] += 1

    # print accuracy for each class
    for classname, correct_count in accuracy_matrix.items():
        true_positive = 100 * float(correct_count[0]) / total_pred[classname] if total_pred[classname] != 0 else 0
        true_negative = 100 * float(correct_count[1]) / total_pred[classname] if total_pred[classname] != 0 else 0
        false_positive = 100 * float(correct_count[2]) / total_pred[classname] if total_pred[classname] != 0 else 0
        false_negative = 100 * float(correct_count[3]) / total_pred[classname] if total_pred[classname] != 0 else 0
        # print(f'True-positive for class: {classname:5s} is {true_positive:.1f} %')
        # print(f'True-negative for class: {classname:5s} is {true_negative:.1f} %')
        # print(f'False-positive for class: {classname:5s} is {false_negative:.1f} %')
        # print(f'False-negative for class: {classname:5s} is {false_positive:.1f} %')

        print(f'Matrix for class: {classname:5s} is\n'
              f' {true_positive:.3f} %'
              f' {true_negative:.3f} %\n'
              f' {false_positive:.3f} %'
              f' {false_negative:.3f} %'
              )
    return accuracy_matrix, total_pred


def get_datasets():
    pass


def generate_folds():
    pass


def get_loaders(trainset, testset, validationset, method='baseline', batch_size=512 // 8):
    sum_of_weigths = len(trainset)
    if method == 'baseline':
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  sampler=RandomSampler(trainset, num_samples=int(sum_of_weigths ),
                                                                        replacement=True))
    elif method == 'oversampling':
        train_samples = [len(list(filter(lambda el: el[-1] == i, trainset.dataset))) for i in
                         range(trainset.classes_n())]
        train_samples = [sum_of_weigths / i for i in train_samples]
        weights = [train_samples[datarow[-1]] for datarow in trainset.dataset]

        sampler = WeightedRandomSampler(weights, num_samples=int(sum_of_weigths ), replacement=True)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=sampler)
    else:
        raise NotImplementedError()

    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset))
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=len(validationset))
    # shuffle=False, num_workers=2)
    return trainloader, testloader, validationloader


def get_network(samples_n, in_dim, out_dim, alpha=3.0):
    hidden_neurons = samples_n / (alpha * (in_dim + out_dim))
    ratios = [0.56, 0.44]
    hidden_layers = [int(ratios[0] * hidden_neurons), int(ratios[1] * hidden_neurons)]
    print(hidden_layers)
    return Net(in_dim, out_dim, hidden_layers)


def save_running_stats():
    pass


def save_end_stats():
    pass


def run_experiment(dataset, folds_number=5):
    generated_sets = dataset.generate_sets(0.05, folds_number)

    for trainset, validationset, testset in generated_sets:
        run_validation_fold(trainset, validationset, testset)


def run_validation_fold(trainset, validationset, testset):
    classes = [key for key, value in sorted(trainset.classes.classes.items(), key=operator.itemgetter(1))]

    net = get_network(len(trainset), trainset.attribute_n(), trainset.classes_n(), 2.5)

    # functions to show an image

    # get some random training images

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    trainloader, testloader, validationloader = get_loaders(trainset, testset, validationset, method='oversampling')
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i:.5f}')
        if epoch % 1 == 0:
            print_summary(net, testloader)
        if running_loss < 0.2 / i:
            break

    print('Finished Training')

    print_detailed_summary(net, testloader, classes)


def main():
    transforms = torch.FloatTensor

    dataset = KEELDataset('./data/yeast.dat', transforms)  # , transform)

    run_experiment(dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)


if __name__ == "__main__":
    main()
