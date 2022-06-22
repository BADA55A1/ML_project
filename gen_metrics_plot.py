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
	print(i + 1, datasets_list[i])

print('Select dataset (0=all): ', end = '')

choice = int(input())

if choice != 0:
	datasets_list = [datasets_list[choice - 1]]

print('Show or save to png? (1 for show): ', end = '')
show = False
if int(input()) == 1:
	show = True

for dataset in datasets_list:
	fig, axs = plt.subplots(2)
	fig.set_figheight(8)  
	fig.set_figwidth(8)

	m_i = 0
	for f in os.listdir('./out'):
		if f.startswith('metric_' + dataset + '_'):
			method = f.split('.')[0].split('_')[-1]
			# print(f, method)

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
				# axs[m_i].set_aspect(60)

				m_i += 1

	fig.suptitle(dataset, fontsize=16)
	plt.tight_layout()

	if show:
		plt.show()
	else:
		if not os.path.exists('./plots'):
			os.makedirs('./plots')
		fig.savefig('./plots/loss_' + dataset + '.png', dpi=300, orientation='portrait')