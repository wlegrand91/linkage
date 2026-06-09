import pandas as pd
import numpy as np

import copy


def sync_cell_and_syringe(cell_contents: dict,
                          syringe_contents: dict) -> tuple[list, list, dict, dict]:
    """
    Reconcile cell and syringe composition dictionaries.

    Ensures both dictionaries contain the same set of species by assigning a
    concentration of 0.0 to any species present in one but absent from the
    other. Also identifies which species are actively titrating (i.e. present
    in the syringe at a concentration different from the cell).

    Parameters
    ----------
    cell_contents : dict
        Macrospecies names mapped to their initial concentrations (mol/L) in
        the cell.
    syringe_contents : dict
        Macrospecies names mapped to their concentrations (mol/L) in the
        syringe.

    Returns
    -------
    species : list
        Sorted list of all species found across both dictionaries.
    titrating_species : list
        Species that are present in the syringe at a concentration different
        from the cell — these are the ones whose cell concentration changes
        over the course of the titration.
    cell_contents : dict
        Copy of the input cell dictionary with zeros filled in for any missing
        species.
    syringe_contents : dict
        Copy of the input syringe dictionary with zeros filled in for any
        missing species.

    Raises
    ------
    ValueError
        If either ``cell_contents`` or ``syringe_contents`` is not a dict.
    """

    if not issubclass(type(cell_contents), dict):
        err = "cell_contents should be a dictionary with initial cell concs\n"
        raise ValueError(err)

    if not issubclass(type(syringe_contents), dict):
        err = "syringe_contents should be a dictionary with initial syringe concs\n"
        raise ValueError(err)

    titrating_species = []
    for s in syringe_contents:
        if s in cell_contents:
            if syringe_contents[s] == cell_contents[s]:
                continue
        titrating_species.append(s)

    species = list(set(syringe_contents.keys()).union(set(cell_contents.keys())))
    species.sort()

    syringe_contents = copy.deepcopy(syringe_contents)
    cell_contents = copy.deepcopy(cell_contents)

    for s in species:
        if s not in syringe_contents:
            syringe_contents[s] = 0.0
        if s not in cell_contents:
            cell_contents[s] = 0.0

    return species, titrating_species, cell_contents, syringe_contents


def _titr_constant_volume(cell_contents: dict,
                          syringe_contents: dict,
                          injection_array: np.ndarray,
                          cell_volume: float,
                          out: dict) -> dict:
    """
    Compute per-injection concentrations under constant-volume conditions.

    Models each injection as: withdraw ``injection_array[i]`` µL from the
    cell, then add the same volume from the syringe, keeping the total cell
    volume fixed at ``cell_volume`` throughout.

    Parameters
    ----------
    cell_contents : dict
        Current cell concentrations (mol/L), updated in-place as injections
        are applied.
    syringe_contents : dict
        Syringe concentrations (mol/L), held constant.
    injection_array : numpy.ndarray
        Volume of each injection in the same units as ``cell_volume``.
    cell_volume : float
        Fixed cell volume.
    out : dict
        Accumulator dictionary mapping species names (and ``'injection'`` and
        ``'volume'``) to lists of values. Populated in-place.

    Returns
    -------
    out : dict
        The updated accumulator dictionary.
    """

    for i in range(len(injection_array)):

        out["injection"].append(injection_array[i])
        out["volume"].append(cell_volume)

        for s in cell_contents.keys():
            prev_conc = cell_contents[s]
            a = (cell_volume - injection_array[i]) * prev_conc
            b = injection_array[i] * syringe_contents[s]
            cell_contents[s] = (a + b) / cell_volume
            out[s].append(cell_contents[s])

    return out


def _titr_increase_volume(cell_contents: dict,
                          syringe_contents: dict,
                          injection_array: np.ndarray,
                          cell_volume: float,
                          out: dict) -> dict:
    """
    Compute per-injection concentrations under increasing-volume conditions.

    Models standard ITC behaviour: each injection adds ``injection_array[i]``
    µL to the cell, increasing the total volume cumulatively.

    Parameters
    ----------
    cell_contents : dict
        Current cell concentrations (mol/L), updated in-place as injections
        are applied.
    syringe_contents : dict
        Syringe concentrations (mol/L), held constant.
    injection_array : numpy.ndarray
        Volume of each injection in the same units as ``cell_volume``.
    cell_volume : float
        Initial cell volume before any injections.
    out : dict
        Accumulator dictionary mapping species names (and ``'injection'`` and
        ``'volume'``) to lists of values. Populated in-place.

    Returns
    -------
    out : dict
        The updated accumulator dictionary.
    """

    current_volume = cell_volume
    for i in range(len(injection_array)):

        new_volume = current_volume + injection_array[i]

        out["injection"].append(injection_array[i])
        out["volume"].append(new_volume)

        for s in cell_contents.keys():
            a = current_volume * cell_contents[s]
            b = injection_array[i] * syringe_contents[s]
            cell_contents[s] = (a + b) / new_volume
            out[s].append(cell_contents[s])

        current_volume = new_volume

    return out


def titrator(cell_contents: dict,
             syringe_contents: dict,
             injection_array: np.ndarray,
             cell_volume: float = 201.3,  # MicroCal ITC200 cell volume (µL)
             constant_volume: bool = False) -> pd.DataFrame:
    """
    Simulate a titration and return per-injection total concentrations.

    Computes the total concentration of every macrospecies in the cell after
    each injection, accounting for dilution and mixing. Species absent from
    either dictionary are assumed to have a concentration of 0.

    Parameters
    ----------
    cell_contents : dict
        Macrospecies names mapped to their initial concentrations (mol/L) in
        the cell.
    syringe_contents : dict
        Macrospecies names mapped to their concentrations (mol/L) in the
        syringe.
    injection_array : numpy.ndarray
        Ordered injection volumes. If a pre-injection baseline point is
        included, the first entry should be 0.
    cell_volume : float, default 201.3
        Initial cell volume in µL (matching the units of ``injection_array``).
        Default is the MicroCal ITC200 cell volume.
    constant_volume : bool, default False
        If ``True``, use constant-volume mode (withdraw-then-inject). If
        ``False``, use standard ITC mode where each injection increases the
        total cell volume.

    Returns
    -------
    pandas.DataFrame
        One row per injection point. Columns are ``'injection'``, ``'volume'``,
        and one column per macrospecies holding its total concentration
        (mol/L) at that point.
    """

    species, _, cell_contents, syringe_contents = sync_cell_and_syringe(
        cell_contents, syringe_contents)

    injection_array = np.array(injection_array, dtype=float)

    out: dict = {"injection": [], "volume": []}
    for s in species:
        out[s] = []

    if constant_volume:
        out = _titr_constant_volume(cell_contents,
                                    syringe_contents,
                                    injection_array,
                                    cell_volume,
                                    out)
    else:
        out = _titr_increase_volume(cell_contents,
                                    syringe_contents,
                                    injection_array,
                                    cell_volume,
                                    out)

    return pd.DataFrame(out)
