import numpy as np
import theano
from theano import tensor as T
import pandas as pd
import os
import gc
import seaborn as sns
from matplotlib import pyplot as plt
import nltk
import cPickle as pickle
from tqdm import tqdm
from collections import Counter, OrderedDict

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from sklearn.metrics import accuracy_score
import datetime, time, json
from string import punctuation
from tqdm import tqdm
import sys

sys.setrecursionlimit(10000)