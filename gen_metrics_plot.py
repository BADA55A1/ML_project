#!/usr/bin/python

import os, json
import matplotlib.pyplot as plt

datasets_list = []
for f in os.listdir('./out'):
	if f.startswith('metric_'):
		dataset = f.split('_')[1]
		if dataset not in datasets_list:
			datasets_list.append(dataset)
		
for i in range(len(datasets_list)):
	print(i, datasets_list[i])

print('Select dataset: ', end = '')
dataset = datasets_list[int(input())]

fig, axs = plt.subplots(2)

for f in os.listdir('./out'):
	if f.startswith('metric_' + dataset + '_'):
		method = f.split('.')[0].split('_')[2]
		print(f, method)

		with open(os.path.join('./out', f), 'r') as infile:
			data = json.load(infile)
			
			line, = axs[0].plot(data["train"])
			line.set_label(method)

			line, = axs[1].plot(data["valid"])
			line.set_label(method)

			
axs[0].set_ylabel('loss')			
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[0].set_title('Training')
axs[1].set_title('Validation')
axs[0].legend()
axs[1].legend()

fig.suptitle(dataset, fontsize=16)
plt.show()