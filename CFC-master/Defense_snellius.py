import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import random
import kmedoids
from sklearn.decomposition import PCA
from zoopt import Dimension, ValueType, Objective, Parameter, Opt, ExpOpt
import seaborn as sns
import subprocess
import torch

import warnings
warnings.filterwarnings('ignore')

from FairClusteringCodebase.fair_clustering.eval.functions import * #[TO-DO] Write base class and derive metrics from it, temporary eval code

from FairClusteringCodebase.fair_clustering.dataset import ExtendedYaleB, Office31, MNISTUSPS
from FairClusteringCodebase.fair_clustering.algorithm import FairSpectral, FairKCenter, FairletDecomposition, ScalableFairletDecomposition

import matplotlib.pyplot as plt

#os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' #:4096:8


