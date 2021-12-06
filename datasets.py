from torch.utils.data import Dataset


class Simple_BB_Dataset(Dataset):
    def __init__(self, boxes, labels):
        """
        Args:
            boxes (np.array): file with bounding boxes with velocities
            labels (np.array): label file

        """
        self.boxes = boxes
        self.labels = labels

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        '''
        Returns:
            sample (dict): Containing:
                features (np.array): bounding boxes with velocities
                label: bounding box label
        '''
        boxes = self.boxes[idx]
        labels = self.labels[idx]

        sample = {'features': boxes, 'labels': labels}
        return sample
