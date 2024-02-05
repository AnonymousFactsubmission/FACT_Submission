## Code for the FACT 2024 paper: [Reproducibility and Extension of Consensus Robust Fair Clustering]

### Requirements:
```
python-mnist
gdown
kmedoids
pulp
torch
scikit-learn==0.22.2
zoopt
pyckmeans
```

### Instructions:
- For the attack section of the paper, please follow the code in `Attack.ipynb`.
- For the defense section of the paper, please follow the code in `Defense.ipynb`.
- For the white defense mechanism, please follow the code in `White_Box_Defense.ipynb`
- Using the code provided in these notebooks, the experimental results in the paper can be obtained.


### Before running:

- For MNIST-USPS dataset, please download the dataset from https://mega.nz/folder/oHJ2UCoK#r62nRoZ0gH8NXIcgmyWReA, clicking on the one named "MNIST_vs_USPS.mat". After downloading it, it should be moved in the folder in position CFC-master/FairClusteringCodebase/fair_clustering/raw_data/mnist_usps/ .Please, check that the file name is "MNIST_vs_USPS.mat".
- Regarding Pacs dataset, it is already downoladed in "Fair-Clustering-Codebase/fair_clustering/raw_data/pacs/pacs_20_x_20.mat". However, to properly load the dataset, the global directory in the attack and defense files should be modified. Please search for "Change path accordingly to the position of the folder in your computer" in the attack and defense files to have some more detailed instructions.
- Office-31 dataset should be downloaded automatically when loaded. If this is not the case, download:
1) domain_adaptation_features_20110616.tar.gz from https://drive.google.com/u/0/uc?id=0B4IapRTv9pJ1WTVSd2FIcW4wRTA&export=download
2) office31_resnet50.zip from https://wjdcloud.blob.core.windows.net/dataset/office31_resnet50.zip

and move both of them to the folder in position "CFC-master/FairClusteringCodebase/fair_clustering/raw_data/office31/".
- To be able to run correctly the KFC algorithm implemented in the original code, please install Gurobi optimizer https://www.gurobi.com/ 