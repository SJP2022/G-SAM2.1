_in_clusters : ndarray of shape (n_clusters,), dtype=floating
        Placeholder for the sums of the weights of every observation assigned
        to each center.
    center_half_distances : ndarray of shape (n_clusters, n_clusters),             dtype=floating
        Half pairwise distances between centers.
    distance_next_center : ndarray of shape (n_clusters,), dtype=floating
        Distance between each center its closest center.
    upper_bounds : ndarray of shape (n_samples,), dtype=floating
        Upper bound for the distance between each sample and its center,
        updated inplace.
    lower_bounds : ndarray of shape (n_samples, n_clusters), dtype=floating
        Lower bound for the distance between each sample and each center,
        updated inplace.
    labels : ndarray of shape (n_samples,), dtype=int
        labels assignment.
    center_shift : ndarray of shape (n_clusters,), dtype=floating
        Distance between old and new centers.
    n_threads : int
        The number of threads to be used by openmp.
    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm. This is useful especially when calling predict on a
          fitted model.
    """
    ...
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                CHUNK_SIZE: int = 256
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         from abc import ABC, abstractmethod as abstractmethod
from numbers import Integral as Integral, Real as Real
from typing import Any, Callable, ClassVar, Literal, TypeVar

from numpy import ndarray
from numpy.random import RandomState

from .._typing import ArrayLike, Float, Int, MatrixLike
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin
from ..exceptions import ConvergenceWarning as ConvergenceWarning
from ..metrics.pairwise import euclidean_distances as euclidean_distances
from ..utils import check_array as check_array, check_random_state as check_random_state
from ..utils._param_validation import (
    Hidden as Hidden,
    Interval as Interval,
    StrOptions as StrOptions,
    validate_params as validate_params,
)
from ..utils._readonly_array_wrapper import ReadonlyArrayWrapper as ReadonlyArrayWrapper
from ..utils.extmath import row_norms as row_norms, stable_cumsum as stable_cumsum
from ..utils.fixes import threadpool_info as threadpool_info, threadpool_limits as threadpool_limits
from ..utils.sparsefuncs import mean_variance_axis as mean_variance_axis
from ..utils.sparsefuncs_fast import assign_rows_csr as assign_rows_csr
from ..utils.validation import check_is_fitted as check_is_fitted
from ._k_means_common import CHUNK_SIZE as CHUNK_SIZE
from ._k_means_elkan import (
    elkan_iter_chunked_dense as elkan_iter_chunked_dense,
    elkan_iter_chunked_sparse as elkan_iter_chunked_sparse,
    init_bounds_dense as init_bounds_dense,
    init_bounds_sparse as init_bounds_sparse,
)
from ._k_means_lloyd import (
    lloyd_iter_chunked_dense as lloyd_iter_chunked_dense,
    lloyd_iter_chunked_sparse as lloyd_iter_chunked_sparse,
)

KMeans_Self = TypeVar("KMeans_Self", bound="KMeans")
MiniBatchKMeans_Self = TypeVar("MiniBatchKMeans_Self", bound="MiniBatchKMeans")

import warnings

import numpy as np
import scipy.sparse as sp

###############################################################################
# Initialization heuristic

def kmeans_plusplus(
    X: MatrixLike | ArrayLike,
    n_clusters: Int,
    *,
    x_squared_norms: None | ArrayLike = None,
    random_state: None | RandomState | int = None,
    n_local_trials: None | Int = None,
) -> tuple[ndarray, ndarray]: ...
def k_means(
    X: MatrixLike | ArrayLike,
    n_clusters: Int,
    *,
    sample_weight: None | ArrayLike = None,
    init: MatrixLike | Callable | Literal["k-means++", "random", "k-means++"] = "k-means++",
    n_init: Literal["auto", "warn"] | int = "warn",
    max_iter: Int = 300,
    verbose: bool = False,
    tol: Float = 1e-4,
    random_state: RandomState | None | Int = None,
    copy_x: bool = True,
    algorithm: Literal["lloyd", "elkan", "auto", "full", "lloyd"] = "lloyd",
    return_n_iter: bool = False,
) -> tuple[ndarray, ndarray, float] | tuple[ndarray, ndarray, float, int]: ...

class _BaseKMeans(ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC):
    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_clusters: Int,
        *,
        init,
        n_init,
        max_iter,
        tol,
        verbose,
        random_state,
    ) -> None: ...
    def fit_predict(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> ndarray: ...
    def predict(self, X: MatrixLike | ArrayLike, sample_weight: None | ArrayLike = None) -> ndarray: ...
    def fit_transform(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> ndarray: ...
    def transform(self, X: MatrixLike | ArrayLike) -> ndarray: ...
    def score(
        self,
        X: MatrixLike | ArrayLike,
        y: Any = None,
        sample_weight: None | ArrayLike = None,
    ) -> float: ...

class KMeans(_BaseKMeans):
    feature_names_in_: ndarray = ...
    n_features_in_: int = ...
    n_iter_: int = ...
    inertia_: float = ...
    labels_: ndarray = ...
    cluster_centers_: ndarray = ...

    _parameter_constraints: ClassVar[dict] = ...

    def __init__(
        self,
        n_clusters: Int = 8,
        *,
        init: MatrixLike | Callable | Literal["k-means++", "random", "k-means++"] = "k-means++",
        n_init: Literal["auto", "warn"] | int = "warn",
        max_iter: Int = 300,
        tol: Float = 1e-4,
        verbose: Int = 0,
        random_state: RandomState | None | Int = None,
        copy_x: bool = True,
        algorithm: Literal["lloyd", "elkan