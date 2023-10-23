#import plotly.express as px
#import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
import json
import os
import sys
print(sys.prefix)
import matplotlib.pyplot as plt
from K_Nearest_Neighbor_Dynamic_Time_Warping import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots

plt.style.use('bmh')

# Extrae datos del archivo json creado por el programa de medición
def procesar(pathtofile):
	
	datos = json.load(open(pathtofile))
	mem_df= pd.json_normalize(datos["resultados"]["datos"]).apply(pd.to_numeric)
		
	x = mem_df['acu.t'].values/1000
	y = mem_df['dat.pre'].values*-.0046

	fase_all = mem_df['acu.F1'].values
	
	idx_max = argrelextrema(y, np.greater,order=3)[0]
	idx_min = argrelextrema(y, np.less,order=3)[0]
	
	if idx_max[0] > idx_min[0]:
		primeromax = True
	else:
		primeromax = False
	
	ymax = [y[i] for i in idx_max]
	ymin = [y[i] for i in idx_min]
	
	if len(ymax)>len(ymin):
	  ymax = ymax[:len(ymin)]
	 
	if len(ymin)>len(ymax):
	  ymin = ymin[:len(ymax)]
	 	
	if primeromax:
		t = [x[i] for i in idx_max]
		fase = [fase_all[i] for i in idx_max]
	else:
		t = [x[i] for i in idx_min]
		fase = [fase_all[i] for i in idx_min]
	
	return fase
		
	
path="/home/dario/Documents/analisis_de_fase"
ts_train = []
ts_train_filenames = []
ts_train_labels = []
ts_test = []
ts_test_labels = []
ts_test_filenames = []

for root, dirs, files in os.walk(path):
		for file in files:
			if os.path.splitext(file)[1] == '.txt':
				filepath = os.path.join(root, file)
				filename = os.path.splitext(file)[0]
				# Las mediciones que incluyen estas secuencias de caracteres no se consideran en el análisis:
				if all(keyword not in filename for keyword in ['PD', 'CC', 'TEST', 'FREC']):
					label = 0 if "mala" in filepath else 1 if "buena" in filepath else -1
					if "test" in filepath:
						print('\nTest: ' + filename)
						ts_test.append(procesar(filepath))
						ts_test_labels.append(label)
						ts_test_filenames.append(filename)
					elif "train" in filepath:				
						ts_train.append(procesar(filepath))
						ts_train_labels.append(label)
						ts_train_filenames.append(filename)
					
					print('\n' + filename+" - label: "+str(label))
					
					
# Find the minimum length lmin among all the vectors in timeseries
lmin_train = min(len(ts) for ts in ts_train)
lmin_test = min(len(ts) for ts in ts_test)
lmin = min(lmin_train,lmin_test)
# Truncate all vectors in A to have a length of lmin
ts_train = [ts[:lmin] for ts in ts_train]
ts_test = [ts[:lmin] for ts in ts_test]
print("lmin: ",lmin)

ts_train = np.array(ts_train)
#ts_train = (ts_train - np.mean(ts_train)) / np.std(ts_train)

ts_train_labels = np.array(ts_train_labels)

ts_test = np.array(ts_test)
#ts_test = (ts_test - np.mean(ts_test)) / np.std(ts_test)

labels = {0:'MALA', 1:'BUENA'}
          
# m = KnnDtw()
# distance = m._dtw_distance(ts_train[0], ts_train[9])

# fig = plt.figure(figsize=(12,4))
# #_ = plt.plot(ts_1, label='ts_1')

# for ts in ts_train:
    # #label = f'ts_{i + 1}'  # Generate a label for the legend
    # plt.plot(ts)


# _ = plt.title('DTW distance between A and B is %.2f' % distance)
# _ = plt.ylabel('Amplitude')
# _ = plt.xlabel('Time')
# #_ = plt.legend()
# plt.show()

nlabels = 2
m = KnnDtw(n_neighbors=3, max_warping_window=100)
m.fit(ts_train, ts_train_labels)
label, proba = m.predict(ts_test)
print('label',label)

for i in range(len(ts_test)):
    print(f'{ts_test_filenames[i]}, real: {ts_test_labels[i]}, fiteado: {label[i]}')

from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(label, ts_test_labels, target_names=[l for l in labels.values()]))

conf_mat = confusion_matrix(label, ts_test_labels)

fig = plt.figure(figsize=(6,6))
width = np.shape(conf_mat)[1]
height = np.shape(conf_mat)[0]

res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
for i, row in enumerate(conf_mat):
    for j, c in enumerate(row):
        if c>0:
            plt.text(j-.2, i+.1, c, fontsize=16)
            
cb = fig.colorbar(res)
plt.title('Confusion Matrix')
_ = plt.xticks(range(nlabels), [l for l in labels.values()], rotation=90)
_ = plt.yticks(range(nlabels), [l for l in labels.values()])

#plt.show()

############
#fig2 = plt.figure(figsize=(12,4))
#_ = plt.plot(ts_1, label='ts_1')

# Define a colormap for mapping labels to colors
colormap = plt.cm.get_cmap('viridis', max(ts_train_labels) + 2)  # Adjust 'viridis' to your preferred colormap

max_label = max(ts_train_labels)
colors = [colormap(i / max_label) for i in ts_train_labels]

fig2, ax = plt.subplots(figsize=(12, 4))

fig0= go.Figure()

for i, ts in enumerate(ts_train):
    label = f'Train file: {ts_train_filenames[i]} Label: {ts_train_labels[i]}'  # Generate a label for the legend

    # Set the color based on the label
    color = colormap(ts_train_labels[i])

    # Plot the time series with the specified color
    ax.plot(ts, label=label, c=color)
    
    fig0.add_traces(go.Scatter(y=ts, name=filename+"_"+str(ts_train_labels[i])))

for i, ts in enumerate(ts_test):
    label = f'Test File: {ts_test_filenames[i]}'  # Generate a label for the legend

    # Set the color based on the label
    color = colormap(3)

    # Plot the time series with the specified color
    ax.plot(ts, label=label, c=color)



fig0.show()

_ = plt.ylabel('Amplitude')
_ = plt.xlabel('Time')
_ = plt.legend()
plt.show()

