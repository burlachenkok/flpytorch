import os
from PIL import Image
from ..h5_tff_dataset import H5TFFDataset


class FLCifar100(H5TFFDataset):
    """
    CIFAR100 Dataset.
    500 clients that were allocated data_preprocess using LDA.
    """
    def __init__(self, h5_path, transform, client_id=None, train=True):
        if train:
            h5_path = os.path.join(h5_path, 'cifar100_fl/fed_cifar100_train.h5')
        else:
            h5_path = os.path.join(h5_path, 'cifar100_fl/fed_cifar100_test.h5')

        super(FLCifar100, self).__init__(h5_path, client_id, 'cifar100_fl', 'image')
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of item that is fetched on behalf on current setup client

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        client, i = self._get_item_preprocess(index)
        img = Image.fromarray(self.dataset[client]['image'][i])
        x = self.transform(img)
        y = self.dataset[client]['label'][i]
        return x, y
