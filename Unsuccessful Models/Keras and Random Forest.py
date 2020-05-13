import tensorflow as tf

    
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(356,activation='relu',kernel_regularizer=tf.keras.regularizers.l1(0.01),input_shape=(X_train.shape[1],)))
#model.add(layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.l1(0.01)))
model.add(tf.keras.layers.Dense(1))
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=opt,loss='mse'
,metrics=[tf.keras.metrics.RootMeanSquaredError()])
hist=model.fit(X_train,y_train,epochs=100,verbose=0,validation_data=(X_valid,y_valid))
epochs = 100
mae=hist.history['root_mean_squared_error']
plt.plot(range(1,epochs),mae[1:],label='mae')
plt.xlabel('epochs')
plt.ylabel('mse')
mae=hist.history['val_root_mean_squared_error']
plt.plot(range(1,epochs),mae[1:],label='val_mae')
plt.legend()






from sklearn.metrics import mean_squared_error

def rmse(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))
def print_rf_score(model):
    print(f'Train R2:   {model.score(X_train, y_train)}')
    print(f'Valid R2:   {model.score(X_valid, y_valid)}')
    print(f'Train RMSE: {rmse(model.predict(X_train), y_train)}')
    print(f'Valid RMSE: {rmse(model.predict(X_valid), y_valid)}')
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs = -1, random_state = 42)
rf.fit(X_train, y_train)
print_rf_score(rf)
