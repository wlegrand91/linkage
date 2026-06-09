import numpy as np


class ExperimentalPoint:
    """
    Base class for a single experimental observation point.

    Stores shared references to the concentration arrays owned by
    ``GlobalModel`` so that every point always reflects the current
    model state without holding its own copy of the data.  Subclass
    this to implement a specific observable type (e.g. ITC heat or
    spectroscopic signal).

    See the README in this directory for the full subclassing contract.
    """

    def __init__(self,
                 idx: int,
                 expt_idx: int,
                 obs_key: str,
                 micro_array: np.ndarray,
                 macro_array: np.ndarray,
                 del_macro_array: np.ndarray,
                 total_volume: float,
                 injection_volume: float):
        """
        Store references to the shared concentration arrays for this point.

        Parameters
        ----------
        idx : int
            Row index of this point within the experiment's data array.
        expt_idx : int
            Index of the parent experiment in ``GlobalModel._expt_list``.
        obs_key : str
            Column name of the observable in the experiment's DataFrame.
        micro_array : numpy.ndarray, shape (n_points, n_micro)
            Shared array of micro-species concentrations.  Filled in-place
            by ``GlobalModel.model()`` — do not copy.
        macro_array : numpy.ndarray, shape (n_points, n_macro)
            Shared array of total macro-species concentrations.
        del_macro_array : numpy.ndarray, shape (n_points, n_macro)
            Shared array of (syringe − cell) concentration differences at
            each injection point.
        total_volume : float
            Total cell volume (L) at this injection point.
        injection_volume : float
            Volume (L) of the injection that produced this point.
        """

        self._idx = idx
        self._expt_idx = expt_idx
        self._obs_key = obs_key
        self._micro_array = micro_array
        self._macro_array = macro_array
        self._del_macro_array = del_macro_array
        self._total_volume = total_volume
        self._injection_volume = injection_volume

    @property
    def idx(self) -> int:
        """Row index of this point within the experiment's data array."""
        return self._idx

    @property
    def expt_idx(self) -> int:
        """Index of the parent experiment in ``GlobalModel._expt_list``."""
        return self._expt_idx

    @property
    def obs_key(self) -> str:
        """Column name of the observable in the experiment's DataFrame."""
        return self._obs_key
