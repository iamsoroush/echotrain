import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append('../echotrain/')

from .evaluator import Evaluator
from .evaluator_echonet import EvaluatorEchoNet
