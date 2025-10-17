import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

with open("30_sequences.pkl", "rb") as f:
    raw_data = pickle.load(f)

X = [np.array(seq) for seq, label in raw_data]
y = [label for seq, label in raw_data]

unique_labels = sorted(set(y))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = [label_map[label] for label in y]

y_cat = to_categorical(y_encoded)

X = np.array(X)
y_cat = np.array(y_cat)
print(f"X shape: {X.shape} — [samples, sequence_length, features]")
print(f"y shape: {y_cat.shape} — [samples, num_classes]")

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded)

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    verbose=1
)

model.save("new_model.h5")


with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

reverse_label_map = {v: k for k, v in label_map.items()}


# Accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Model Accuracy')
plt.show()

# Loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Model Loss')
plt.show()