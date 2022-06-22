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

m_i = 0
for f in os.listdir('./out'):
	if f.startswith('metric_' + dataset + '_'):
		method = f.split('.')[0].split('_')[2]
		print(f, method)

		with open(os.path.join('./out', f), 'r') as infile:
			data = json.load(infile)
			
			line, = axs[m_i].plot(data["train"])
			line.set_label('Training')

			line, = axs[m_i].plot(data["valid"])
			line.set_label('Validation')

			
			axs[m_i].set_ylabel('loss')
			axs[m_i].set_xlabel('epoch')
			axs[m_i].set_title(method)
			axs[m_i].legend()

			m_i += 1

fig.suptitle(dataset, fontsize=16)
plt.show()