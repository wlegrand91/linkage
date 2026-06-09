# Observation Point Classes

This directory contains the point classes used by `GlobalModel` to represent
individual experimental observations and connect them to the thermodynamic
model.  Each point stores references to the shared concentration arrays for
its experiment and implements the logic for computing its predicted value and
analytical derivatives.

Two concrete implementations are provided: `ITCPoint` (injection heats) and
`SpecPoint` (spectroscopic signal).  If you need to support a new observable
type, create a new subclass of `ExperimentalPoint` following the contract
described here.

---

## Base class: `ExperimentalPoint`

All point classes must subclass `ExperimentalPoint`.  The base class stores
the shared arrays that `GlobalModel` fills in during `model()`, so that every
point always has access to the current concentration state without holding its
own copy of the data.

### Constructor

Call `super().__init__()` with the following arguments:

| Argument | Type | Description |
|---|---|---|
| `idx` | `int` | Row index of this point within the experiment's data array |
| `expt_idx` | `int` | Index of the parent experiment in `GlobalModel._expt_list` |
| `obs_key` | `str` | Column name of the observable in the experiment's DataFrame |
| `micro_array` | `np.ndarray (n_points × n_micro)` | Shared array of micro-species concentrations; filled by `GlobalModel.model()` |
| `macro_array` | `np.ndarray (n_points × n_macro)` | Shared array of total macro-species concentrations |
| `del_macro_array` | `np.ndarray (n_points × n_macro)` | Shared array of (syringe − cell) concentration differences at each point |
| `total_volume` | `float` | Total cell volume (L) at this injection point |
| `injection_volume` | `float` | Volume (L) of the injection that produced this point |

`micro_array`, `macro_array`, and `del_macro_array` are **shared references**
to the arrays owned by `GlobalModel`.  Do not copy them — the point class
accesses them by reference so that `GlobalModel` can update concentrations
in-place and all points see the current state automatically.

Access the stored values via `self._idx`, `self._micro_array`, etc.

---

## Required methods

### `calc_value(self, parameters, **kwargs) -> float`

Return the predicted value of this observable given the current concentration
state and parameter vector.

- `parameters` is the full regression parameter vector (numpy array) as
  managed by `GlobalModel`.
- Read concentrations from `self._micro_array[self._idx, :]` and
  `self._macro_array[self._idx, :]`.
- For observables that depend on the *change* between injections (e.g. ITC
  heats), access the previous point via `self._micro_array[self._idx - 1, :]`
  and guard against `self._idx == 0`.
- Use `**kwargs` to accept extra arguments passed by `GlobalModel` (e.g.
  `full_dh_array` for ITC).  Ignore what you don't need.

---

### `get_d_y_d_concs(self) -> np.ndarray`

Return `d(y_calc) / d(micro_species_concs)` as a 1-D array of length
`n_micro`.  This is used by `GlobalModel.jacobian_normalized()` to propagate
the concentration Jacobian into the observation Jacobian via the chain rule:

```
d(y) / d(params) = get_d_y_d_concs() @ d(concs) / d(params)
```

- If the observable is a simple function of the current micro-species (e.g.
  a spectroscopic fraction), compute the exact derivative.
- If the Jacobian for this observable type is handled entirely inside
  `GlobalModel.jacobian_normalized()` (as with ITC, where the heat depends
  on concentration *differences* across two shots), return
  `np.zeros(n_micro)` and document that the full derivative logic lives in
  `GlobalModel`.

---

### `get_d_y_d_other_params(self, parameters, **kwargs) -> dict`

Return derivatives of this observable with respect to any parameters that
are **not** binding constants — i.e. parameters that don't affect
micro-species concentrations through the binding model.  The return value
is a dict mapping **parameter index** (integer position in the full
`parameters` vector) to the scalar derivative value.

- Return an empty dict `{}` if there are no such direct dependencies.
- For ITC, this covers heats of dilution (`nuisance_dil_*` parameters).
- `GlobalModel.jacobian_normalized()` writes these values directly into the
  Jacobian matrix after computing the concentration chain-rule terms.

---

## Registering a new point type in `GlobalModel`

Once your subclass is implemented:

1. Import it at the top of `global_model.py`.
2. In `GlobalModel._add_point()`, add a branch that detects your new
   observable type (via `obs_info["type"]`), constructs your point with
   any extra kwargs it needs, and appends it to `self._points`.
3. In `GlobalModel.model()`, add a branch in the `for pt in self._points`
   loop if your point requires extra arguments to `calc_value` (like
   `full_dh_array` for ITC).
4. In `GlobalModel.jacobian_normalized()`, handle the Jacobian assembly for
   your observable if `get_d_y_d_concs()` is insufficient (i.e. the
   derivative depends on more than just the current concentration vector).
5. Register the new type string (e.g. `"mytype"`) in
   `Experiment.define_<mytype>_observable()` and add it to the
   `_define_generic_observable` validation if needed.

---

## Summary of the two existing point types

| Class | `obs_info["type"]` | `calc_value` returns | `get_d_y_d_concs` | `get_d_y_d_other_params` |
|---|---|---|---|---|
| `SpecPoint` | `"spec"` | `sum(micro_num) / macro_denom` | `obs_mask / denom` (exact) | `{}` |
| `ITCPoint` | `"itc"` | injection heat (cal) | `zeros` — full derivative in `GlobalModel` | dilution heat derivatives |
