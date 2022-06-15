#!/usr/bin/python

import os
from torch.utils.data import Dataset
# import zipfile

class KEELDatasetAttribute:
    def __init__(self, line):
        self.name = None
        self.type = None
        self.min = None
        self.max = None

        if line.startswith('@attribute'):
            self.name = line.split(' ')[1].strip()

            type_str = line.split(' ')[2]
            if type_str == 'integer':
                self.type = int
            elif type_str == 'real':
                self.type = float
            elif type_str.startswith('{0,'):
                self.type = bool
            else:
                raise Exception('unknown @attribute type: ' + line)

            if self.type is not bool:
                limits_str = line.split('[', 1)[1].split(']')[0]
                limits_str = limits_str.split(',')
                self.min = self.type(limits_str[0].strip())
                self.max = self.type(limits_str[1].strip())
        else:
            raise Exception('not an @attribute string: ' + line)

    def print(self):
        print(self.name, self.type, self.min, self.max)

    def convert(self, input):
        out = self.type(input)
        
        if self.type is not bool:
            if out < self.min or out > self.max:
                raise Exception(
                    'Value ' + input +
                    ' for attribute "' + self.name + 
                    '" is out of limits [' + str(self.min) +
                    ', ' + str(self.max) + ']'
                )

        return out


class KEELDatasetClass:
    def __init__(self, line):
        classes_str = line.split('{', 1)[1].split('}')[0]
        classes = classes_str.split(',')

        self.classes = {}
        self.classes_n = len(classes)
        self.classes_representation = [0] * self.classes_n

        for i in range(self.classes_n):
            self.classes[classes[i].strip()] = i

    def convert(self, value):
        class_index = self.classes[value.strip()]
        self.classes_representation[class_index] += 1
        return class_index



class KEELDataset(Dataset):
    def __init__(self, data_path):
        self.name = None
        self.attributes = []
        self.dataset = []
        self.classes = None

        with open(data_path, "r") as datastream:
            data_started = False
            for line in datastream:

                # Header reading:
                if not data_started:
                    if line.startswith('@relation'):
                        self.name = line.split(' ')[1].strip()

                    elif line.startswith('@attribute'):
                        if line.split(' ')[1].strip().lower() == 'class':
                            self.classes = KEELDatasetClass(line.strip())
                        else:
                            self.attributes.append(KEELDatasetAttribute(line.strip()))

                    elif line.startswith('@data'):
                        data_started = True

                    elif line.startswith('@inputs') or line.startswith('@outputs') or line.startswith('@input') or line.startswith('@output'):
                        pass
                    else:
                        print('unknown header line: ' + line)
                        # raise Exception('unknown header line: ' + line)

                # Data reading:
                else:
                    data = line.strip().split(',')

                    sample = []
                    
                    for i in range(len(data) - 1):
                        sample.append(
                            self.attributes[i].convert(data[i])
                        )

                    sample.append(
                        self.classes.convert(
                            data[len(data) - 1]
                        )
                    )

                    self.dataset.append(sample)
                  
        # Checking:  
    def check(self):
        print('attribute_n:', self.attribute_n())
        print('inputs_metadata:', self.inputs_metadata())
        print('classes_n:', self.classes_n())
        print('classes_names:', self.classes_names())
        print('classes_representation:', self.classes_representation())
        print('__len__:', self.__len__())

        for attribute in self.attributes:
            attribute.print()
        
        sample_len_t = self.attribute_n() + 1
        for sample in self.dataset:
            if len(sample) != sample_len_t:
                raise Exception(
                        'Wrong data length for sample "' + str(sample) + 
                        ', expected ' + str(sample_len_t)
                )

    def attribute_n(self):
        return len(self.attributes)

    def inputs_metadata(self):
        return self.attributes

    def classes_n(self):
        return self.classes.classes_n

    def classes_names(self):
        return list(self.classes.classes)

    def classes_representation(self):
        return self.classes.classes_representation

    def getitem(self, idx):
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx][:-1]
        label = self.dataset[idx][-1]
        return data, label
        # data = self.dataset[idx]
        # if self.transform:
        #     data = self.transform(data)

        # label = self.dataset_class[idx]
        # if self.target_transform:
        #     label = self.target_transform(label)

        # return data, label


# Test
dataset = KEELDataset('data/yeast1.dat')
# dataset.check()
print(dataset.getitem(5) )

# Parse all data files
data_files = [os.path.join('./data', f) for f in os.listdir('./data') if os.path.isfile(os.path.join('./data', f))]
for file_path in data_files:
    print(file_path)
    dataset = KEELDataset(file_path)

