# Обучение модели

# import numpy as np
# import tensorflow as tf

# # Загружаем данные из файла

# def write_to_historical_data(data):
#     try:
#         with open("historical_data.txt", "a") as file:
#             file.write(data + "\n")
#         print("Data has been successfully written to historical_data.txt")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Take input from the user up to 10 times
# for i in range(1):
#     data_to_write = input(f"Enter data {i+1}")
#     write_to_historical_data(data_to_write)

# def load_data(file_path):
#     with open(file_path, "r") as file:
#         data = file.readlines()
#     return [int(d.strip()) for d in data]

# # Генерируем обучающие примеры и метки
# def generate_sequences(data, seq_length):
#     sequences = []
#     labels = []
#     for i in range(len(data) - seq_length):
#         sequences.append(data[i:i + seq_length])
#         labels.append(data[i + seq_length])
#     return np.array(sequences), np.array(labels)

# # Создаем LSTM модель
# def create_model(input_shape):
#     model = tf.keras.Sequential([
#         tf.keras.layers.LSTM(64, input_shape=input_shape),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#     return model

# # Загружаем и предобрабатываем данные
# data = load_data("historical_data.txt")
# sequence_length = 10
# X, y = generate_sequences(data, sequence_length)

# # Разделяем данные на обучающую и тестовую выборки (90% - обучение, 10% - тест)
# split_index = int(0.9 * len(X))
# X_train, X_test = X[:split_index], X[split_index:]
# y_train, y_test = y[:split_index], y[split_index:]

# # Преобразуем данные в формат, подходящий для обучения LSTM
# X_train = X_train.reshape(-1, sequence_length, 1)
# X_test = X_test.reshape(-1, sequence_length, 1)

# # Создаем и компилируем модель
# model = create_model(input_shape=(sequence_length, 1))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Обучаем модель
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# # Оцениваем модель на тестовой выборке
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test accuracy:", accuracy)

# # Предсказываем следующее значение на основе исторических данных
# last_sequence = data[-sequence_length:]
# next_value = model.predict(np.array([last_sequence]))[0][0]
# next_value = round(next_value)  # Округляем до ближайшего целого числа
# print("Predicted next value:", int(next_value))
# model.save('./models')





# Работа уже с обученной моделю

import numpy as np
import tensorflow as tf

# Загружаем данные из файла

def write_to_historical_data(data):
    try:
        with open("historical_data.txt", "a") as file:
            file.write(data + "\n")
        print("Data has been successfully written to historical_data.txt")
    except Exception as e:
        print(f"An error occurred: {e}")

# Take input from the user up to 10 times
for i in range(1):
    data_to_write = input(f"Enter data {i+1} (e.g., 2023-08-05, 10.2, 25.6, 35.8): ")
    write_to_historical_data(data_to_write)

def load_data(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()
    return [int(d.strip()) for d in data]

# Генерируем обучающие примеры и метки
def generate_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

# Загружаем сохраненную модель
loaded_model = tf.keras.models.load_model('./models')

# Загружаем и предобрабатываем данные
data = load_data("historical_data.txt")
sequence_length = 10
X, y = generate_sequences(data, sequence_length)

# Разделяем данные на обучающую и тестовую выборки (90% - обучение, 10% - тест)
split_index = int(0.9 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Преобразуем данные в формат, подходящий для обучения LSTM
X_train = X_train.reshape(-1, sequence_length, 1)
X_test = X_test.reshape(-1, sequence_length, 1)

# Компилируем загруженную модель1
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Оцениваем модель на тестовой выборке
loss, accuracy = loaded_model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

# Предсказываем следующее значение на основе исторических данных
last_sequence = data[-sequence_length:]
next_value = loaded_model.predict(np.array([last_sequence]))[0][0]
next_value = round(next_value)  # Округляем до ближайшего целого числа
print("Predicted next value:", int(next_value))
