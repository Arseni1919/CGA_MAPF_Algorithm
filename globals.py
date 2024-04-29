import logging
import random
import math
import abc
import copy
from functools import lru_cache, reduce
from collections import OrderedDict
import itertools
from itertools import combinations, permutations, tee
from collections import defaultdict, Counter, namedtuple, deque
from datetime import datetime
import time
import json
from os.path import exists
import cProfile
import pstats
import re
import heapq
import os
import operator
import concurrent.futures
from pprint import pprint
from typing import *

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# from scipy.spatial.distance import cdist
# import plotly.graph_objects as go
# import plotly.express as px
# import pandas as pd
# import neptune.new as neptune
# import streamlit as st
# import plotly.express as px
# import plotly.figure_factory as ff
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import graphviz
# import torch
# import torchvision
# import torchvision.transforms as T
# from torchvision.io import ImageReadMode

AgentTuple = namedtuple('AgentTuple', ['num', 'start_node_name', 'curr_node_name', 'next_goal_node_name', 'arrived'])
AgentTupleMAPF = namedtuple('AgentTupleMAPF', ['num', 'start_node_name', 'goal_node_name'])

markers = ['-^', '-1', '-2', '-X', '-d', '-v', '-o']
markers_iter = iter(markers)
markers_lines_dict = defaultdict(lambda: next(markers_iter))
colors_dict: DefaultDict[str, str | None] = defaultdict(lambda: None)


markers_lines_dict['LNS2'] = '-p'
colors_dict['LNS2'] = 'blue'

markers_lines_dict['PF-LNS2'] = '-*'
colors_dict['PF-LNS2'] = 'red'

markers_lines_dict['PrP'] = '-v'
colors_dict['PrP'] = 'green'

markers_lines_dict['PF-PrP'] = '-^'
colors_dict['PF-PrP'] = 'orange'

color_names = [
    # 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w',  # Single-letter abbreviations
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white',  # Full names
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black',
    'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
    'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod',
    'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid',
    'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite',
    'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow',
    'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
    'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen',
    'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite',
    'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown',
    'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet',
    'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]

















