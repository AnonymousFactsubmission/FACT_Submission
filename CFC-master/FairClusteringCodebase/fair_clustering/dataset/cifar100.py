import os
import sys
import numpy as np
import scipy.io
from torchvision.datasets import CIFAR100
from torchvision import transforms
import torch
from fair_clustering.dataset import ImageDataset


class Cifar100(ImageDataset):
    """ https://faculty.cc.gatech.edu/~judy/domainadapt/ """

    dataset_name = "CIFAR100"
    dataset_dir = os.path.join(sys.path[1], "fair_clustering/raw_data/cifar100")
    dataset_dir = "fair_clustering/raw_data/cifar100"
    file_url = {
        "cifar-100-python.tar.gz": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    }
    all_domain = ["omnivore", "herbivore", "carnivore", "not applicable"]
    groups = {0: 1,
     1: 0,
     2: 0,
     3: 0,
     4: 1,
     5: 3,
     6: 1,
     7: 1,
     8: 3,
     9: 3,
     10: 3,
     11: 0,
     12: 3,
     13: 3,
     14: 1,
     15: 1,
     16: 3,
     17: 3,
     18: 1,
     19: 1,
     20: 3,
     21: 0,
     22: 3,
     23: 3,
     24: 0,
     25: 3,
     26: 0,
     27: 2,
     28: 3,
     29: 0,
     30: 2,
     31: 1,
     32: 2,
     33: 3,
     34: 0,
     35: 0,
     36: 0,
     37: 3,
     38: 1,
     39: 3,
     40: 3,
     41: 3,
     42: 2,
     43: 2,
     44: 0,
     45: 0,
     46: 0,
     47: 3,
     48: 3,
     49: 3,
     50: 0,
     51: 3,
     52: 3,
     53: 1,
     54: 3,
     55: 2,
     56: 3,
     57: 1,
     58: 3,
     59: 3,
     60: 3,
     61: 3,
     62: 3,
     63: 1,
     64: 0,
     65: 1,
     66: 0,
     67: 2,
     68: 3,
     69: 3,
     70: 3,
     71: 3,
     72: 2,
     73: 2,
     74: 0,
     75: 0,
     76: 3,
     77: 1,
     78: 2,
     79: 2,
     80: 0,
     81: 3,
     82: 3,
     83: 3,
     84: 3,
     85: 3,
     86: 3,
     87: 3,
     88: 2,
     89: 3,
     90: 3,
     91: 2,
     92: 3,
     93: 1,
     94: 3,
     95: 2,
     96: 3,
     97: 2,
     98: 0,
     99: 0}
    label = {
        'apple': 0,
         'aquarium_fish': 1,
         'baby': 2,
         'bear': 3,
         'beaver': 4,
         'bed': 5,
         'bee': 6,
         'beetle': 7,
         'bicycle': 8,
         'bottle': 9,
         'bowl': 10,
         'boy': 11,
         'bridge': 12,
         'bus': 13,
         'butterfly': 14,
         'camel': 15,
         'can': 16,
         'castle': 17,
         'caterpillar': 18,
         'cattle': 19,
         'chair': 20,
         'chimpanzee': 21,
         'clock': 22,
         'cloud': 23,
         'cockroach': 24,
         'couch': 25,
         'crab': 26,
         'crocodile': 27,
         'cup': 28,
         'dinosaur': 29,
         'dolphin': 30,
         'elephant': 31,
         'flatfish': 32,
         'forest': 33,
         'fox': 34,
         'girl': 35,
         'hamster': 36,
         'house': 37,
         'kangaroo': 38,
         'keyboard': 39,
         'lamp': 40,
         'lawn_mower': 41,
         'leopard': 42,
         'lion': 43,
         'lizard': 44,
         'lobster': 45,
         'man': 46,
         'maple_tree': 47,
         'motorcycle': 48,
         'mountain': 49,
         'mouse': 50,
         'mushroom': 51,
         'oak_tree': 52,
         'orange': 53,
         'orchid': 54,
         'otter': 55,
         'palm_tree': 56,
         'pear': 57,
         'pickup_truck': 58,
         'pine_tree': 59,
         'plain': 60,
         'plate': 61,
         'poppy': 62,
         'porcupine': 63,
         'possum': 64,
         'rabbit': 65,
         'raccoon': 66,
         'ray': 67,
         'road': 68,
         'rocket': 69,
         'rose': 70,
         'sea': 71,
         'seal': 72,
         'shark': 73,
         'shrew': 74,
         'skunk': 75,
         'skyscraper': 76,
         'snail': 77,
         'snake': 78,
         'spider': 79,
         'squirrel': 80,
         'streetcar': 81,
         'sunflower': 82,
         'sweet_pepper': 83,
         'table': 84,
         'tank': 85,
         'telephone': 86,
         'television': 87,
         'tiger': 88,
         'tractor': 89,
         'train': 90,
         'trout': 91,
         'tulip': 92,
         'turtle': 93,
         'wardrobe': 94,
         'whale': 95,
         'willow_tree': 96,
         'wolf': 97,
         'woman': 98,
         'worm': 99
    }

    def __init__(self, exclude_domain: str, download=True, center=True, use_feature=False):
        assert exclude_domain in self.all_domain, "Exclude domain for %s dataset should be %s" % (
            self.dataset_name, " or ".join([s for s in self.all_domain]))
        self.download_or_check_data(self.dataset_dir, self.file_url, download, use_gdown=True)

        domains = self.all_domain
        domains.remove(exclude_domain)
        transform = transforms.Compose([
                                transforms.ToTensor()
                                ])
        if not use_feature:
            s = []
        #     X, y, s = [], [], []
        #     for i, domain in enumerate(domains):
        #         domain_dir = os.path.join(self.dataset_dir, "domain_adaptation_features", domain, "interest_points")
        #         category = os.listdir(domain_dir)
        #         for c in category:
        #             img_paths = os.listdir(os.path.join(domain_dir, c))
        #             img_paths = [os.path.join(domain_dir, c, p) for p in img_paths if p[-17:-14] == "800"]
        #             imgs = [np.asarray(scipy.io.loadmat(p)["histogram"]).reshape(-1) for p in img_paths]

        #             X.extend(imgs)
        #             y.extend([self.label[c] for _ in range(len(imgs))])
        #             s.extend([i for _ in range(len(imgs))])

        #     X, y, s = np.asarray(X), np.asarray(y), np.asarray(s)
        # else:
        #     domain_1 = np.genfromtxt(
        #         os.path.join(self.dataset_dir, "resnet50_feature", "%s_%s.csv" % (domains[0], domains[0])),
        #         delimiter=",")
        #     domain_2 = np.genfromtxt(
        #         os.path.join(self.dataset_dir, "resnet50_feature", "%s_%s.csv" % (domains[1], domains[1])),
        #         delimiter=",")

        #     X = np.concatenate([domain_1[:, :-1], domain_2[:, :-1]], axis=0)
        #     y = np.concatenate([domain_1[:, -1], domain_2[:, -1]], axis=0)
        #     s = np.concatenate([np.zeros(domain_1.shape[0]), np.ones(domain_2.shape[0])], axis=0)
            cifar100 = CIFAR100(root=self.dataset_dir, train=True, transform=transform, download=True)
            data_loader = torch.utils.data.DataLoader(cifar100, batch_size=1, shuffle=True)
            X, y = [],[]
            for image,label in data_loader:
            
                X.append(image.numpy())
                y.append(label.item())
                s.append(self.groups[label.item()])
            X, y, s = np.asarray(X), np.asarray(y), np.asarray(s)
            X = X.reshape(X.shape[0], -1)
        super(Cifar100, self).__init__(
            X=X,
            y=y,
            s=s,
            center=center,
        )


if __name__ == "__main__":
    dataset = Cifar100(exclude_domain="not applicable", use_feature=True)
    X, y, s = dataset.data
    stat = dataset.stat
