from zipfile import ZipFile
import os
import pandas as pd
import keras
import matplotlib.pyplot as plt
 
uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"
 
df = pd.read_csv(csv_path)
 
feature_keys = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]
 
t_data = df[feature_keys[0]]
 
plt.plot(t_data)
plt.show()



######## PASITAISYTI !!!!


def get_model():
    model = Sequential()
    model.add(Dense(units = 16, activation='relu'))
    model.add(Dense(units = 16, activation='relu'))
    model.add(Dense(units = 16, activation='relu'))
    model.add(Dense(units = 1, activation='tanh'))

    # metrics = acc_val
    model.compile(optimizer='Adam',loss='mse', metrics=['mae'])
    return model

NumberofSamples = 5000
t = np.linspace(0,5, NumberofSamples)
y = np.sin(2*np.pi *t) + np.random.randn(NumberofSamples) * 0.01

#
#y_norm = (y - np.min(y))/ (np.max(y) -np.min(y) )
y_norm = ((y - np.mean(y)) / np.std(y)   )

# 1 is added for one output
NumberOfImputs = 3 + 1
xTrain = list()
yTrain = list()
for i in range(0, NumberofSamples - NumberOfImputs):
    xTrain.append(y[i:i+NumberOfImputs-1])
    yTrain.append(y[i + NumberOfImputs])

xTrain = np.asarray(xTrain,dtype=np.float)
yTrain = np.asarray(yTrain,dtype=np.float)


model = get_model()
model.fit(x=xTrain, y=yTrain, batch_size=32, epochs=20, verbose=1)

# get forecasted values.
forecasted = model.predict(xTrain)
#forecasted = forecasted.sqeezed

print('len xTrain:{}'.format(xTrain.shape))
print('len yTrain:{}'.format(yTrain.shape))


plt.plot(t, y,'r.-', mfc='black',mec = 'black')
plt.plot(t[NumberOfImputs:],forecasted,'b-')
plt.grid(b=True)
plt.show()