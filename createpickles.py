from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import keras.optimizers as kop
import numpy as np
from sklearn.preprocessing import StandardScaler
try:
    import cPickle as pickle
except:
    import pickle

X_train = np.array([[1., 1.], [1., 0], [0, 1.], [0, 0]])
y_train = np.array([[0.], [1.], [1.], [0.]])

scalar_x = StandardScaler()
X_train = scalar_x.fit_transform(X_train)

scalar_y = StandardScaler()
y_train = scalar_y.fit_transform(y_train)

print 'dumping StandardScaler objects ..'
pickle.dump(scalar_y,
            open('pickles/scalar_y.pickle', "wb"),
            protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(scalar_x,
            open('pickles/scalar_x.pickle', "wb"),
            protocol=pickle.HIGHEST_PROTOCOL)

xin = X_train.shape[1]

model = Sequential()
model.add(Dense(output_dim=4, input_shape=(xin, )))
model.add(Activation('tanh'))
model.add(Dense(4))
model.add(Activation('linear'))
model.add(Dense(1))

rms = kop.RMSprop()

print 'compiling now..'
model.compile(loss='mse', optimizer=rms)

model.fit(X_train, y_train, nb_epoch=1000, batch_size=1, verbose=2)
score = model.evaluate(X_train, y_train, batch_size=1)

open('pickles/my_model_architecture.json', 'w').write(model.to_json())
model.save_weights('pickles/my_model_weights.h5')
