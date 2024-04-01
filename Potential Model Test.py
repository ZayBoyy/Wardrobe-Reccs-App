import tensorflow as tf 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input,Conv2D, Dense, Flatten ,Dropout,MaxPooling2D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

fashion_mnist=tf.keras.datasets.fashion_mnist

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

X_train=X_train/255
X_test=X_test/255

X_train=np.expand_dims(X_train,-1) # Expanding by 1 because the images are in greyscale
X_test=np.expand_dims(X_test,-1)
print(X_train.shape)

num_classes=len(set(y_train))
print('Number of classes: ',num_classes)

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.summary()



model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

history=model.fit(X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=10,
    batch_size=32)

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

report = classification_report(y_true, y_pred)
print(report)

# Defining class names for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Generating confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plotting confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', square=True,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

misclassified_indices = np.where(y_pred != y_true)[0]

num_rows = 5
num_cols = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
for i, idx in enumerate(misclassified_indices[:num_rows * num_cols]):
    img = X_test[idx]
    pred_label = class_names[y_pred[idx]]
    true_label = class_names[y_true[idx]]
    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(img, cmap='gray')
    axes[row, col].set_title(f'Pred: {pred_label}\nTrue: {true_label}')
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()

