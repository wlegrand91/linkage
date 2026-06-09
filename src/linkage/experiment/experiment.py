from linkage.experiment.titrator import titrator
from linkage.experiment.titrator import sync_cell_and_syringe

import numpy as np
import pandas as pd

import copy
import warnings


def _load_dataframe(expt_data: str | pd.DataFrame) -> pd.DataFrame:
    """
    Load experimental data into a DataFrame.

    Accepts either a file path (CSV, TSV, XLS, XLSX) or an existing DataFrame.
    If a file path is given, the delimiter is inferred from the extension; for
    unknown extensions a delimiter is guessed automatically.

    Parameters
    ----------
    expt_data : str or pandas.DataFrame
        Path to a spreadsheet file or a DataFrame containing experimental data.

    Returns
    -------
    pandas.DataFrame
        Loaded data as a DataFrame.

    Raises
    ------
    ValueError
        If ``expt_data`` is neither a string nor a DataFrame.
    """

    if type(expt_data) is str:

        filename = expt_data
        ext = filename.split(".")[-1].strip().lower()

        if ext in ["xlsx", "xls"]:
            df = pd.read_excel(filename)
        elif ext == "csv":
            df = pd.read_csv(filename, sep=",")
        elif ext == "tsv":
            df = pd.read_csv(filename, sep="\t")
        else:
            df = pd.read_csv(filename, sep=None, engine="python")

    elif type(expt_data) is pd.DataFrame:
        df = expt_data.copy()

    else:
        err = f"\n\n'expt_data' {expt_data} not recognized. Should be the\n"
        err += "filename of a spreadsheet or a pandas dataframe.\n"
        raise ValueError(err)

    return df


def _preprocess_df(expt_data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize a raw experimental DataFrame.

    Ensures the DataFrame has an ``injection`` column and an ``ignore_point``
    column (added as all-False if absent). If the first row does not correspond
    to a zero injection (i.e. a pre-injection baseline measurement), a synthetic
    row with ``injection=0`` and ``ignore_point=True`` is prepended so that
    concentration tracking starts from the correct initial state.

    Parameters
    ----------
    expt_data : pandas.DataFrame
        Raw experimental data, must contain an ``injection`` column.

    Returns
    -------
    pandas.DataFrame
        Preprocessed DataFrame with guaranteed ``injection`` and
        ``ignore_point`` columns, and a zero-injection leading row if needed.

    Raises
    ------
    ValueError
        If ``expt_data`` does not contain an ``injection`` column.
    """

    if "injection" not in expt_data.columns:
        err = "expt_data should be a dataframe or spreadsheet with a 'injection' column\n"
        raise ValueError(err)

    if "ignore_point" not in expt_data.columns:
        expt_data["ignore_point"] = np.zeros(len(expt_data), dtype=bool)

    if not np.isclose(expt_data.loc[expt_data.index[0], "injection"], 0):
        new_row = dict([(c, [np.nan]) for c in expt_data.columns])
        new_row["injection"] = [0.0]
        new_row["ignore_point"] = [True]
        new_row = pd.DataFrame(new_row)
        expt_data = pd.concat((new_row, expt_data), ignore_index=True)

    return expt_data


class Experiment:

    def __init__(self,
                 expt_data: str | pd.DataFrame,
                 cell_contents: dict,
                 syringe_contents: dict,
                 cell_volume: float,
                 conc_to_float: str | None = None,
                 constant_volume: bool = False):
        """
        Load an experimental dataset and compute per-injection concentrations.

        Reads injection data and initial cell/syringe compositions, then uses
        the titrator to compute the total concentration of every macrospecies
        at each injection point. Observables (ITC heat or spectroscopic signal)
        are registered separately via ``define_itc_observable`` or
        ``define_spectroscopic_observable``.

        Parameters
        ----------
        expt_data : str or pandas.DataFrame
            Experimental data as a file path or DataFrame. Must contain an
            ``injection`` column with injection volumes. An optional
            ``ignore_point`` boolean column can mark points to exclude from
            fitting; any unlabelled points with NaN observables are also
            excluded automatically.
        cell_contents : dict
            Macrospecies names mapped to their concentrations (mol/L) in the
            cell before any injection.
        syringe_contents : dict
            Macrospecies names mapped to their concentrations (mol/L) in the
            syringe. Species absent from either dict are assigned a
            concentration of 0.
        cell_volume : float
            Volume of the cell in the same units as the ``injection`` column
            (typically µL).
        conc_to_float : str, optional
            Name of the macrospecies whose concentration is treated as
            uncertain (e.g. a macromolecule whose pipetted amount is imprecise).
            When set, the fitter will include a multiplicative ``fudge`` factor
            for this species' concentration.
        constant_volume : bool, default False
            If ``True``, each injection is modelled as "withdraw X µL then add
            X µL from syringe", keeping the cell volume constant. If ``False``
            (standard ITC mode), each injection increases the total cell volume.

        Raises
        ------
        ValueError
            If ``conc_to_float`` names a species not present in the computed
            concentration table.
        """

        expt_data = _load_dataframe(expt_data=expt_data)
        self._expt_data = _preprocess_df(expt_data=expt_data)

        out = sync_cell_and_syringe(cell_contents, syringe_contents)
        _, titrating_species, cell_contents, syringe_contents = out
        self._initial_cell_contents = copy.deepcopy(cell_contents)
        self._syringe_contents = copy.deepcopy(syringe_contents)
        self._titrating_macro_species = copy.deepcopy(titrating_species)

        self._expt_concs = titrator(cell_contents=cell_contents,
                                    syringe_contents=syringe_contents,
                                    injection_array=np.array(self._expt_data["injection"]),
                                    cell_volume=cell_volume,
                                    constant_volume=constant_volume)

        if conc_to_float is not None:
            if conc_to_float not in self._expt_concs.columns:
                err = "conc_to_float is not a macrospecies in the experiment\n"
                raise ValueError(err)

        self._conc_to_float = conc_to_float
        self._observables = {}

    def _define_generic_observable(self,
                                   obs_column: str,
                                   obs_std: str | float) -> tuple[str, str]:
        """
        Validate observable column and standard deviation, returning column names.

        Checks that ``obs_column`` exists in the experimental data and is not
        the ``injection`` column. If ``obs_std`` is a float, a new column named
        ``{obs_column}_std`` is created and populated with that constant value.
        Any rows where the observable is NaN are flagged as ``ignore_point``.

        Parameters
        ----------
        obs_column : str
            Name of the column in ``expt_data`` holding the observable values.
        obs_std : str or float
            Either the name of an existing column holding per-point standard
            deviations, or a single float applied uniformly to all points.

        Returns
        -------
        obs_column : str
            Validated observable column name (unchanged).
        obs_std_column : str
            Name of the standard deviation column (existing or newly created).

        Raises
        ------
        ValueError
            If ``obs_column`` is not found in the data, equals ``'injection'``,
            or if ``obs_std`` cannot be interpreted as a column name or float.
        """

        if obs_column not in self._expt_data.columns:
            err = f"obs_column '{obs_column}' should be one of the columns in the experimental data\n"
            raise ValueError(err)

        if obs_column == "injection":
            err = "obs_column cannot be 'injection'\n"
            raise ValueError(err)

        if obs_std in self._expt_data.columns:
            obs_std_column = obs_std
        else:
            obs_std_column = f"{obs_column}_std"

            try:
                obs_std = float(obs_std)
            except Exception as e:
                err = "obs_std should be either the name of a column or a single value\n"
                raise ValueError(err) from e

            self._expt_data.loc[:, obs_std_column] = obs_std

        if obs_column in self._observables:
            w = f"obs_column '{obs_column}' was already added. Overwriting\n"
            warnings.warn(w)

        set_to_ignore = np.isnan(self._expt_data[obs_column])
        self._expt_data.loc[set_to_ignore, "ignore_point"] = True

        return obs_column, obs_std_column

    def define_itc_observable(self,
                              obs_column: str,
                              obs_std: str | float) -> None:
        """
        Register an ITC heat observable for this experiment.

        The named column is expected to contain integrated injection heats
        (kcal/mol of injectant, or equivalent). These are used directly as the
        model target during fitting.

        Parameters
        ----------
        obs_column : str
            Name of the column in the experimental data holding injection heats.
        obs_std : str or float
            Per-point uncertainty. Either the name of a column in the
            experimental data, or a single float used for all points.
        """

        obs_column, obs_std_column = self._define_generic_observable(
            obs_column=obs_column,
            obs_std=obs_std)

        self._observables[obs_column] = {"type": "itc",
                                         "std_column": obs_std_column}

    def define_spectroscopic_observable(self,
                                        obs_column: str,
                                        obs_std: str | float,
                                        obs_microspecies: list[str] | str,
                                        obs_macrospecies: str) -> None:
        """
        Register a spectroscopic observable for this experiment.

        The observable is modelled as the fraction of a macrospecies that
        exists in the specified microspecies states::

            signal = sum(microspecies) / macrospecies_total

        Parameters
        ----------
        obs_column : str
            Name of the column in the experimental data holding the
            spectroscopic signal.
        obs_std : str or float
            Per-point uncertainty. Either the name of a column in the
            experimental data, or a single float used for all points.
        obs_microspecies : list of str or str
            Microscopic species from the binding model that contribute to the
            observed signal. A single string is also accepted.
        obs_macrospecies : str
            Macroscopic species used as the denominator (total concentration
            normaliser). Must be present in ``cell_contents`` or
            ``syringe_contents``.

        Raises
        ------
        ValueError
            If ``obs_microspecies`` is not iterable, or if ``obs_macrospecies``
            is not a recognised macrospecies in the experiment.

        Examples
        --------
        For a reaction where the signal of ``A`` changes upon binding ``B``::

            AT = A + AB
            BT = B + AB

        set ``obs_microspecies=["AB"]`` and ``obs_macrospecies="AT"``.
        """

        obs_column, obs_std_column = self._define_generic_observable(
            obs_column=obs_column,
            obs_std=obs_std)

        if issubclass(type(obs_microspecies), str):
            obs_microspecies = [obs_microspecies]

        if not hasattr(obs_microspecies, '__iter__'):
            err = "obs_microspecies should be a list of species contributing to signal\n"
            raise ValueError(err)

        expt_macrospecies = self._expt_concs.columns
        expt_macrospecies = [m for m in expt_macrospecies if m != "injection"]
        if obs_macrospecies not in expt_macrospecies:
            err = f"obs_macrospecies must be one of: {','.join(expt_macrospecies)}\n"
            raise ValueError(err)

        self._observables[obs_column] = {"type": "spec",
                                         "std_column": obs_std_column,
                                         "microspecies": obs_microspecies,
                                         "macrospecies": obs_macrospecies}

    def add_expt_conc_column(self,
                             new_column: str,
                             conc_vector: np.ndarray | None = None) -> None:
        """
        Add a concentration column to the per-injection concentration table.

        Used to inject externally computed or model-defined concentration
        columns (e.g. a nuisance species not tracked by the titrator) into
        ``expt_concs``. If the column already exists it is left unchanged.

        Parameters
        ----------
        new_column : str
            Name of the new column to add.
        conc_vector : numpy.ndarray, optional
            Concentration values, one per injection point. If not provided,
            the column is initialised to zeros.
        """

        if conc_vector is None:
            conc_vector = np.zeros(len(self._expt_concs))

        if new_column not in self._expt_concs.columns:
            self._expt_concs[new_column] = conc_vector

    @property
    def expt_data(self) -> pd.DataFrame:
        """Experimental data DataFrame, including injection volumes and observables."""
        return self._expt_data

    @property
    def expt_concs(self) -> pd.DataFrame:
        """Per-injection total concentrations of all macrospecies in the cell."""
        return self._expt_concs

    @property
    def observables(self) -> dict:
        """Registered observables keyed by column name."""
        return self._observables

    @property
    def conc_to_float(self) -> str | None:
        """Macrospecies whose concentration is allowed to float in the fit, or None."""
        return self._conc_to_float

    @property
    def syringe_contents(self) -> dict:
        """Initial syringe concentrations (mol/L) keyed by macrospecies name."""
        return self._syringe_contents

    @property
    def initial_cell_contents(self) -> dict:
        """Initial cell concentrations (mol/L) keyed by macrospecies name."""
        return self._initial_cell_contents

    @property
    def titrating_macro_species(self) -> list:
        """Macrospecies that change concentration over the titration."""
        return self._titrating_macro_species
