#!/usr/bin/python

# import os
from torch.utils.data import Dataset
# import zipfile

class KEELDatasetAttribute:
    def __init__(self, line):
        self.name = None
        self.type = None
        self.min = None
        self.max = None
        self.is_output = False

        if line.startswith('@attribute'):
            self.name = line.split(' ')[1].strip()

            if self.name == 'Class':
                self.type = bool
                self.is_output = True
            
            else:
                type_str = line.split(' ')[2]
                if type_str == 'integer':
                    self.type = int
                elif type_str == 'real':
                    self.type = float
                else:
                    raise Exception('unknown @attribute type: ' + line)

                limits_str = line.split('[', 1)[1].split(']')[0]
                limits_str = limits_str.split(',')
                self.min = self.type(limits_str[0].strip())
                self.max = self.type(limits_str[1].strip())
        else:
            raise Exception('not an @attribute string: ' + line)

    def print(self):
        print(self.name, self.type, self.min, self.max, self.is_output)

    def convert(self, input):
        if self.type is bool:
            return input == 'positive'
        else:
            out = self.type(input)
            if out < self.min or out > self.max:
                raise Exception(
                    'Value ' + input +
                    ' for attribute "' + self.name + 
                    '" is out of limits [' + str(self.min) +
                    ', ' + str(self.max) + ']'
                )
            return out

    def is_output(self):
        return self._is_output


class KEELDataset(Dataset):
    def __init__(self, data_path):
        self.name = None
        self.dataset = {}
        self.attributes = {}

        with open(data_path, "r") as datastream:
            data_started = False
            for line in datastream:

                # Header reading:
                if not data_started:
                    if line.startswith('@relation'):
                        self.name = line.split(' ')[1].strip()
                    elif line.startswith('@attribute'):
                        attribute = KEELDatasetAttribute(line)
                        self.attributes[attribute.name] = attribute
                        self.dataset[attribute.name] = []
                    elif line.startswith('@data'):
                        data_started = True
                    else:
                        print('unknown header line: ' + line)
                        # raise Exception('unknown header line: ' + line)

                # Data reading:
                else:
                    attributes = list(self.attributes.values() )
                    data = line.strip().split(',')

                    for i in range(len(data) ):
                        self.dataset[attributes[i].name].append(
                            attributes[i].convert(data[i])
                        )
        # Checking:
        for attribute in self.attributes:
            self.attributes[attribute].print()
        
        data_len_t = len(self)
        for attribute in self.dataset:
            if len(self.dataset[attribute]) != data_len_t:
                raise Exception(
                        'data length for attribute "' + attribute + 
                        '" is ' + str(len(self.dataset[attribute]) ) +
                        ', expected ' + str(data_len_t)
                    )
        print('dataset length:', data_len_t)


    def __len__(self):
        return len(
            self.dataset[
                list(self.attributes)[0]
            ]
        )

    def __getitem__(self, idx):
        pass
        # data = self.dataset[idx]
        # if self.transform:
        #     data = self.transform(data)

        # label = self.dataset_class[idx]
        # if self.target_transform:
        #     label = self.target_transform(label)

        # return data, label

dataset = KEELDataset('.tmp/iris0.dat')


'''
    def __init__(self, root, pack_zip_file, dataset_dir, test, transform=None, target_transform=None,
                 cross_validation_k_nr=1):
        self.transform, self.target_transform = transform, target_transform

        cwd = os.path.dirname(__file__)
        cwd_root = os.path.join(cwd, root)
        pack_file = pack_zip_file[:-4]
        cross_validation_k_nr = cross_validation_k_nr

        if pack_file not in os.listdir(cwd_root):
            with zipfile.ZipFile(os.path.join(cwd_root, pack_zip_file), 'r') as zip_ref:
                zip_ref.extractall(cwd_root)

        pack_rel_path = os.path.join(cwd_root, pack_file)

        dataset_dir_rel_path = os.path.join(pack_rel_path, dataset_dir)

        if dataset_dir + '-5-fold' not in os.listdir(dataset_dir_rel_path):
            with zipfile.ZipFile(os.path.join(dataset_dir_rel_path, dataset_dir + '-5-fold.zip'),
                                 'r') as zip_ref:  # todo not working for rar files
                os.mkdir(os.path.join(dataset_dir_rel_path, dataset_dir))
                zip_ref.extractall(os.path.join(dataset_dir_rel_path, dataset_dir))

        self.dataset_metadata_path = os.path.join(dataset_dir_rel_path, dataset_dir + '-5-fold',
                                                  dataset_dir + f'-names.txt')
        if test:
            self.dataset_path = os.path.join(dataset_dir_rel_path, dataset_dir + '-5-fold',
                                             dataset_dir + f'-5-{cross_validation_k_nr}tst.dat')
        else:
            self.dataset_path = os.path.join(dataset_dir_rel_path, dataset_dir + '-5-fold',
                                             dataset_dir + f'-5-{cross_validation_k_nr}tra.dat')

        # TODO load data, formats ect. , best this should be lazy
        self.dataset = []
        self.dataset_class = []
        # this are not necessary for network,
        self.class_to_text_labels = {}
        self.inputs_to_text_labels = {}
        self.inputs_dimentions = {}


dataset = KEELDataset('data', 'imb_IRhigherThan9p1.zip', 'yeast6', test=False)

'''

