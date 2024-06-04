import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# MinMaxScaler 정의 -> data를 0부터 1사이의 값으로 변환(normalize)
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# Train Parameter
seq_length = 7  # 7일 단위로 학습시키고 8일째를 예측
input_dim = 14
hidden_dim = 14
output_dim = 1  # 강수량
learning_rate = 0.01
iterations = 5001

# Data load
xy = np.loadtxt('DataSet/2016 일간날씨+무 평균가격(태백).csv', delimiter=',')
xy = MinMaxScaler(xy)  # Normalize
x = xy
y = xy[:, [-1]]

# Build dataset
# seq_length 만큼을 x, 그다음을 y 반복
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i: i + seq_length]
    _y = y[i + seq_length]
    dataX.append(_x)
    dataY.append(_y)

# Train/test set 나누기
train_size = int(len(dataY) * 0.9)  # train size = 90%
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

# Build the GRU model using Keras
model = tf.keras.Sequential([
     tf.keras.layers.LSTM(hidden_dim, input_shape=(seq_length, input_dim), return_sequences = True), 
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.LSTM(hidden_dim, return_sequences = False), 
     tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(output_dim, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='mean_squared_error')

# Train the model
history = model.fit(trainX, trainY, epochs=iterations//100, batch_size=1, verbose=2)

# Make predictions
predictArray = model.predict(testX)

# Calculate RMSE
rmse_val = np.sqrt(np.mean((testY - predictArray) ** 2))
print("RMSE: {}".format(rmse_val))

# Plot predictions
plt.figure(1)
plt.plot(testY, color="red")
plt.plot(predictArray, color="blue")
plt.xlabel("Time Period")
plt.ylabel("average temperature")
plt.show()
