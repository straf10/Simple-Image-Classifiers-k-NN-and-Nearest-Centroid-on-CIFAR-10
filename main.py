# main.py

import numpy as np
from data_loader import load_cifar10
from knn import KNN, NearestCentroid

# Φόρτωση των δεδομένων
train_images, train_labels, test_images, test_labels = load_cifar10()



# Χρήση υποσυνόλου των δεδομένων
small_train_images = train_images[:1000]
small_train_labels = train_labels[:1000]
small_test_images = test_images[:100]
small_test_labels = test_labels[:100]

# Εκτέλεση με k=1
knn_1 = KNN(k=1)
knn_1.fit(small_train_images, small_train_labels)
predictions_k1 = knn_1.predict(small_test_images)

# Εκτέλεση με k=3
knn_3 = KNN(k=3)
knn_3.fit(small_train_images, small_train_labels)
predictions_k3 = knn_3.predict(small_test_images)

# Υπολογισμός της ακρίβειας για k=1 και k=3
accuracy_k1 = np.mean(np.array(predictions_k1) == small_test_labels)
accuracy_k3 = np.mean(np.array(predictions_k3) == small_test_labels)

# Αρχικοποίηση του Nearest Centroid Classifier
nc = NearestCentroid()
nc.fit(small_train_images, small_train_labels)
predictions_nc = nc.predict(small_test_images)

# Υπολογισμός της ακρίβειας για τον Nearest Centroid
accuracy_nc = np.mean(np.array(predictions_nc) == small_test_labels)

print(f"Accuracy for Nearest Centroid: {accuracy_nc * 100:.2f}%")
print(f"Accuracy for k=1: {accuracy_k1 * 100:.2f}%")
print(f"Accuracy for k=3: {accuracy_k3 * 100:.2f}%")

