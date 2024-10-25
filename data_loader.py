import pickle
import numpy as np
import os

# Συνάρτηση για να φορτώσεις τα δεδομένα από CIFAR-10
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Συνάρτηση για τη φόρτωση των δεδομένων CIFAR-10
def load_cifar10():
    data_path = r'C:\Users\ΝΙΚΟΣ\OneDrive\Υπολογιστής\cifar-10-batches-py'

    # Φόρτωση όλων των batches εκπαίδευσης
    train_images_list = []
    train_labels_list = []

    for i in range(1, 6):
        data_batch = unpickle(os.path.join(data_path, f'data_batch_{i}'))
        train_images_list.append(data_batch[b'data'])
        train_labels_list.append(data_batch[b'labels'])

    # Συγχώνευση όλων των batch δεδομένων
    train_images = np.concatenate(train_images_list)
    train_labels = np.concatenate(train_labels_list)

    # Αναδιάταξη του σχήματος των εικόνων σε (50000, 32, 32, 3)
    train_images = train_images.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)

    # Φόρτωση του test set
    test_batch = unpickle(os.path.join(data_path, 'test_batch'))

    test_images = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # Αναδιάταξη του σχήματος των εικόνων σε (10000, 32, 32, 3)
    test_images = test_images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

    return train_images, train_labels, test_images, test_labels
