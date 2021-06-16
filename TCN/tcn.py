import numpy as np
import tensorflow as tf
from tcn import TCN, tcn_full_summary
import os
from tqdm import tqdm

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# входной слой - векторы по 8 входных значений (дополняются нулями при обучении без контекста)
input_dim = 11
tcn_cells = 128

data = [
    [2, 4, 2, 1, 3, 4, 5, 4],
    [2, 4, 2, 5, 3, 3, 5, 3],
    [2, 5, 4, 5, 5, 5, 5, 4],
    [2, 3, 1, 4, 5, 5, 5, 2],
    [3, 2, 4, 4, 5, 5, 3, 2],
    [2, 1, 3, 1, 3, 5, 2, 1],
    [4, 1, 3, 1, 2, 4, 4, 1],
    [1, 1, 4, 2, 2, 3, 2, 1],
    [2, 1, 4, 1, 2, 3, 4, 1],
    [3, 1, 2, 2, 3, 2, 3, 2],
    [2, 1, 1, 4, 1, 3, 2, 3],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [3, 1, 3, 3, 1, 4, 1, 3],
    [4, 2, 5, 4, 2, 3, 2, 4],
    [3, 2, 1, 5, 3, 3, 2, 4],
    [4, 2, 1, 5, 3, 5, 1, 4],
    [3, 3, 1, 5, 3, 3, 2, 3],
    [3, 2, 2, 5, 1, 2, 3, 3],
    [3, 2, 2, 4, 1, 5, 2, 2],
    [2, 2, 3, 3, 1, 3, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 2, 3, 3, 5, 3, 3],
    [1, 1, 2, 2, 4, 5, 4, 4],
    [2, 1, 2, 2, 4, 3, 5, 2],
    [1, 1, 4, 2, 4, 3, 5, 2],
    [1, 1, 5, 1, 5, 3, 5, 5],
    [2, 1, 5, 1, 5, 2, 3, 3],
    [1, 1, 4, 1, 5, 2, 2, 4],
    [2, 3, 1, 1, 4, 1, 3, 3],
    [1, 5, 2, 1, 3, 2, 1, 4],
    [1, 5, 3, 3, 3, 4, 1, 4],
    [1, 3, 3, 2, 5, 4, 1, 4],
    [2, 4, 3, 2, 5, 2, 2, 3],
    [4, 3, 4, 2, 3, 1, 1, 2],
    [5, 5, 3, 2, 2, 1, 2, 5],
    [5, 5, 4, 1, 1, 1, 5, 2],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 3, 3, 2, 1, 3, 5],
    [4, 2, 3, 5, 2, 2, 4, 4],
    [3, 2, 2, 5, 1, 2, 3, 5],
    [3, 1, 1, 5, 1, 1, 3, 5],
    [3, 1, 1, 3, 1, 3, 4, 5],
    [4, 3, 4, 2, 1, 3, 3, 5],
    [4, 3, 5, 2, 1, 2, 5, 3],
    [4, 4, 5, 2, 3, 1, 5, 3],
    [5, 4, 5, 1, 2, 1, 5, 2],
    [2, 3, 4, 1, 1, 1, 2, 2],
    [2, 2, 4, 1, 1, 4, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 3, 4, 3, 1, 2, 1, 2],
    [2, 3, 5, 4, 3, 1, 2, 3],
    [1, 3, 5, 4, 1, 1, 3, 2],
    [1, 2, 5, 4, 1, 2, 3, 4],
    [3, 4, 5, 5, 4, 4, 2, 3],
    [1, 4, 3, 5, 2, 3, 5, 5],
    [1, 2, 3, 5, 1, 4, 5, 5],
    [1, 1, 4, 4, 1, 4, 5, 4],
]

time_steps = len(data[0])
inputs = tf.keras.Input(shape=(None, 11,))
tcns = []
for idx in range(11):
    lm = tf.keras.layers.Lambda(lambda x, i: x[:, :, i:i + 1], output_shape=(None, None, 1))
    lm.arguments = {'i': idx}
    split = lm(inputs)
    tcn = TCN(name=f'TCN-{idx}', nb_filters=tcn_cells, input_shape=(None, None, 1),
              dilations=[1, 2, 4, 8, 16, 32, 64, 128],
              nb_stacks=1,
              return_sequences=False)(split)
    tcns.append(tcn)
#
concat = tf.keras.layers.Concatenate(axis=1)(tcns)
dense = tf.keras.layers.Dense(64, activation=tf.nn.relu)(concat)
d1 = tf.keras.layers.Dropout(0.1)(dense)
d2 = tf.keras.layers.Dense(16, activation=tf.nn.relu)(d1)
d3 = tf.keras.layers.Dropout(0.1)(d2)
outputs = tf.keras.layers.Dense(8, activation=None)(d3)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mae')
train = []
Y = []

# window = 8
for window in range(1, 24):
    for i in range(window, len(data)):
        Y.append((data[i])[:])
        dx = (data[i - window:i])[:]
        res = []
        for d in dx:
            q = d[:]
            q.extend([0.0, 0.0, 0.0])
            res.append(q)
        train.append(res)

losses = list()
best_loss = None
for epoch_nb in range(50):
    epoch_losses = list()
    for i in tqdm(range(0, len(train))):
        current_loss = model.train_on_batch(np.array([train[i]]), np.array([Y[i]]))
        epoch_losses.append(current_loss)

    avg = sum(epoch_losses) / (1.0 * len(epoch_losses))
    if best_loss is None or best_loss > avg:
        best_loss = avg
        model.save_weights("best.dat")
        print("Best model saved for ", best_loss)

model.load_weights("best.dat")
print("Model loaded")
for i in range(4, len(data)):
    dx = (data[:i])[:]
    res = []
    for d in dx:
        q = d[:]
        q.extend([0.0, 0.0, 0.0])
        res.append(q)
    result = model.predict([res])
    dp = []
    for x in result[0]:
        dp.append(round(x))
    print(dp)
    print(data[i])

# model.fit(train, Y, epochs=100, callbacks=[model_checkpoint_callback])
# model.load_weights(checkpoint_filepath)
# print(model.predict(train))
# print(type(data))
#
# # print(model.layers)
# # print(model.layers[13].receptive_field)
# model.summary()
