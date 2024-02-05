import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import scipy.io

class PACSDataset(Dataset):
    """
    PACS dataset for fair clustering.
    """
    dataset_name = "PACS"

    def __init__(self, root_dir, domain_exclude=None, transform=None, shuffle=True, seed=42):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([transforms.Resize((30, 30))])
        self.image_paths = []  # To store paths to images
        self.label_encodings = {}  # To store label encodings
        self.domain_encodings = {}  # To store domain encodings
        self.labels = []  # To store encoded labels
        self.domains = []  # To store encoded domains
        domain_exclude = set(domain_exclude) if domain_exclude else set()
        random.seed(seed)  # For reproducibility

        # Populate the image paths and encode labels and domains
        for domain in os.listdir(self.root_dir):
            if domain in domain_exclude:
                continue  # Skip excluded domains

            # Encode domain
            if domain not in self.domain_encodings:
                self.domain_encodings[domain] = len(self.domain_encodings)

            domain_path = os.path.join(self.root_dir, domain)
            for label in os.listdir(domain_path):
                # Encode label
                if label not in self.label_encodings:
                    self.label_encodings[label] = len(self.label_encodings)

                label_path = os.path.join(domain_path, label)
                for img_name in os.listdir(label_path):
                    self.image_paths.append(os.path.join(label_path, img_name))
                    self.labels.append(self.label_encodings[label])
                    self.domains.append(self.domain_encodings[domain])

        if shuffle:
            # Shuffle the dataset
            combined = list(zip(self.image_paths, self.labels, self.domains))
            random.shuffle(combined)
            self.image_paths[:], self.labels[:], self.domains[:] = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        domain = self.domains[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, domain

    def save_transformed_data(self, file_path):
        """
        Save the transformed images, labels, and domains to a .mat file.
        """
        X, y, s = self.get_data()
        scipy.io.savemat(file_path, {'X': X, 'y': y, 's': s})

    @staticmethod
    def load_transformed_data(file_path):
        """
        Load transformed images, labels, and domains from a .mat file.
        """
        data = scipy.io.loadmat(file_path)
        return data['X'], data['y'], data['s']


    def get_data(self):
        """
        Loads all images, applies transformations, and returns the dataset as numpy arrays.
        """
        X, y, s = [], [], []

        for idx in range(len(self)):
            image, label, domain = self[idx]
            X.append(image)
            y.append(label)
            s.append(domain)

        # Convert to torch tensors or numpy arrays as needed
        X = torch.stack(X) if isinstance(X[0], torch.Tensor) else np.stack(X)
        X = X.reshape(X.shape[0], -1)

        X = np.array(X)
        y = np.array(y)
        s = np.array(s)
        return X, y, s

