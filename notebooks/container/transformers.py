# Script that includes preliminary re-implementation of some existing
# Transformers in sktime using an ExtensionArray data container

import numpy as np
import pandas as pd

from sktime.transformers.series_as_features.segment import IntervalSegmenter
from sklearn.utils import check_random_state

class RandomIntervalSegmenter(IntervalSegmenter):
    def __init__(self, n_intervals='sqrt', min_length=2, random_state=None):
        self.min_length = min_length
        self.n_intervals = n_intervals
        self.random_state = random_state
        super(RandomIntervalSegmenter, self).__init__()

    def fit(self, X, y=None):

        if not isinstance(self.min_length, int):
            raise ValueError(f"Min_lenght must be an integer, but found: "
                             f"{type(self.min_length)}")
        if self.min_length < 1:
            raise ValueError(f"Min_lenght must be an positive integer (>= 1), "
                             f"but found: {self.min_length}")

        col = X.columns[0]
        X = X[col]
        self.input_shape_ = X.shape
        self._time_index = X.time_index[0, :]

        if self.n_intervals == 'random':
            self.intervals_ = self._rand_intervals_rand_n(self._time_index)
        else:
            self.intervals_ = self._rand_intervals_fixed_n(self._time_index, n_intervals=self.n_intervals)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        colname = X.columns[0]
        X = X[colname]

        # Check inputs.
        self.check_is_fitted()
        slices = [X.slice_time(np.arange(start=a, stop=b)).data for (a, b) in self.intervals_]

        columns = []
        for s, i in zip(slices, self.intervals_):
            # TODO: make sure there are no duplicate names
            columns.append(f"{colname}_{i[0]}_{i[1]}")

        return pd.DataFrame(slices, columns=columns)

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state
        self._rng = check_random_state(random_state)

    def _rand_intervals_rand_n(self, x):
        starts = []
        ends = []
        m = x.shape[0]  # series length
        W = self._rng.randint(1, m, size=int(np.sqrt(m)))
        for w in W:
            size = m - w + 1
            start = self._rng.randint(size, size=int(np.sqrt(size)))
            starts.extend(start)
            for s in start:
                end = s + w
                ends.append(end)
        return np.column_stack([starts, ends])

    def _rand_intervals_fixed_n(self, x, n_intervals):
        len_series = len(x)
        # compute number of random intervals relative to series length (m)
        # TODO use smarter dispatch at construction to avoid evaluating if-statements here each time function is called
        if np.issubdtype(type(n_intervals), np.integer) and (n_intervals >= 1):
            pass
        elif n_intervals == 'sqrt':
            n_intervals = int(np.sqrt(len_series))
        elif n_intervals == 'log':
            n_intervals = int(np.log(len_series))
        elif np.issubdtype(type(n_intervals), np.floating) and (n_intervals > 0) and (n_intervals <= 1):
            n_intervals = int(len_series * n_intervals)
        else:
            raise ValueError(f'Number of intervals must be either "random", "sqrt", a positive integer, or a float '
                             f'value between 0 and 1, but found {n_intervals}.')

        # make sure there is at least one interval
        n_intervals = np.maximum(1, n_intervals)

        starts = self._rng.randint(len_series - self.min_length + 1, size=n_intervals)
        if n_intervals == 1:
            starts = [starts]  # make it an iterable

        ends = [start + self._rng.randint(self.min_length, len_series - start + 1) for start in starts]
        return np.column_stack([starts, ends])


class RandomIntervalFeatureExtractor(RandomIntervalSegmenter):

    def __init__(self, n_intervals='sqrt', min_length=2, features=None,
                 random_state=None):
        super(RandomIntervalFeatureExtractor, self).__init__(
            n_intervals=n_intervals,
            min_length=min_length,
            random_state=random_state,
        )
        self.features = features

    def transform(self, X, y=None):
        colname = X.columns[0]
        X = X[colname]

        # Check is fit had been called
        self.check_is_fitted()

        # Check input of feature calculators, i.e list of functions to be
        # applied to time-series
        if self.features is None:
            features = [np.mean]
        elif isinstance(self.features, list) and all(
                [callable(func) for func in self.features]):
            features = self.features
        else:
            raise ValueError(
                'Features must be list containing only functions (callables) '
                'to be applied to the data columns')

        n_rows, n_columns = X.data.shape
        n_features = len(features)

        n_intervals = len(self.intervals_)

        # Compute features on intervals.
        Xt = np.zeros((n_rows,
                       n_features * n_intervals))  # Allocate output array
        # for transformed data
        columns = []

        i = 0
        for func in features:
            # TODO generalise to series-to-series functions and function kwargs
            for start, end in self.intervals_:
                # Try to use optimised computations over axis if possible,
                # otherwise iterate over rows.
                Xt[:, i] = np.apply_along_axis(func, 1, X.data[:, start:end])
                i += 1
                columns.append(
                    f'{colname}_{start}_{end}_{func.__name__}')

        Xt = pd.DataFrame(Xt)
        Xt.columns = columns
        return Xt

