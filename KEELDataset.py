import os
from torch.utils.data import Dataset
import zipfile



class KEELDataset(Dataset):
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            data = self.transform(data)

        label = self.dataset_class[idx]
        if self.target_transform:
            label = self.target_transform(label)

        return data, label


dataset = KEELDataset('data', 'imb_IRhigherThan9p1.zip', 'yeast6', test=False)
