import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import numpy as np
import matplotlib.pyplot as plt
x=np.random.rand(100,1)
#print(X)
y=2*x+0.05
print(y)

model=Sequential()
model.add(Dense(2,input_dim=1))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001),loss='mean_squared_error')


model.fit(x,y,epochs=500)

output=model.predict(x)
#print(output)


plt.scatter(x,y,label='original data')
plt.plot(x,output,label='predicted data')
plt.show()