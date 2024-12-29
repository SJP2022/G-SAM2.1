import numpy as np
from numpy import ndarray
from scipy.sparse import issparse as issparse

from .._typing import ArrayLike, MatrixLike
from ..base import TransformerMixin
from ..utils.valida