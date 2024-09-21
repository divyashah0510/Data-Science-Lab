# Importing required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# # Loading the MNIST dataset
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Preprocessing data
# x_train = x_train.reshape(-1, 28*28) / 255.0
# x_test = x_test.reshape(-1, 28*28) / 255.0

# # Building the ANN model
# model = models.Sequential([
#     layers.Dense(128, activation='relu', input_shape=(784,)),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

# # Compiling the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Training the model
# history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# # Evaluate the model
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f"Test accuracy: {test_acc:.2f}")


# # Confusion matrix
# y_pred = model.predict(x_test)
# y_pred_classes = y_pred.argmax(axis=1)

# # cm = confusion_matrix(y_test, y_pred_classes)
# # plt.figure(figsize=(8,6))
# # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# # plt.title("Confusion Matrix - MNIST")
# # plt.show()


# # Plotting sample predictions
# plt.figure(figsize=(10,4))
# for i in range(5):
#     plt.subplot(1, 5, i+1)
#     plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
#     plt.title(f"Pred: {y_pred_classes[i]}")
#     plt.axis('off')
# plt.show()


# Loading CIFAR-10 dataset
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# Normalizing data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Building the CNN model
cnn_model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compiling the model
cnn_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Training the model
history = cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = cnn_model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

# Confusion matrix for CIFAR-10
y_pred = cnn_model.predict(x_test)
y_pred_classes = y_pred.argmax(axis=1)

# cm = confusion_matrix(y_test, y_pred_classes)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix - CIFAR-10")
# plt.show()

# Plotting sample predictions
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Pred: {y_pred_classes[i]}")
    plt.axis("off")
plt.show()
