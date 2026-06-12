.. include:: links.rst

==================
Setup and Fitting
==================

This page walks through setting up a linkage model and running a basic fit
from start to finish.

Defining a model spec
=====================

A linkage model is defined by a plain-text specification with up to three
sections: ``equilibria``, ``species``, and an optional ``reparameterize``
block.

**equilibria** lists each chemical reaction and the equilibrium constant that
governs it. Each line has the form ``reactants -> products; constant_name``.

**species** defines conservation relationships. Each line equates a total
concentration (e.g. ``AT``) to a sum of the species that make it up.

**reparameterize** (optional) expresses the physical equilibrium constants in
terms of a smaller set of regression parameters. This is useful when the model
has symmetry constraints or when you want to fit microscopic constants rather
than the raw equilibrium constants.

A minimal example for a single Ca2+/EDTA binding step:

.. code-block:: text

    equilibria:
        E + C -> EC; KE

    species:
        ET = E + EC
        CT = C + EC

A more complex example with a six-state protein and reparameterization:

.. code-block:: text

    equilibria:
        C + E -> EC; KE
        A -> I; KI
        A + C -> AC1; K1
        AC1 + C -> AC2; K2
        AC2 + C -> AC3; K3
        AC3 + C -> AC4; K4

    species:
        ET = E + EC
        AT = I + A + AC1 + AC2 + AC3 + AC4
        CT = C + EC + AC1 + 2*AC2 + 3*AC3 + 4*AC4

    reparameterize:
        K1 = 2 * k_high
        K2 = k_high / 2
        K3 = 2 * k_low
        K4 = k_low / 2
        dH_1 = dH_high
        dH_2 = dH_high
        dH_3 = dH_low
        dH_4 = dH_low

The ``reparameterize`` block here encodes the symmetry of the four binding
sites: two high-affinity and two low-affinity. The fitter sees ``k_high`` and
``k_low`` rather than the four raw constants, reducing the parameter space and
eliminating redundancy.

The spec can be written as a string or saved to a ``.txt`` file and loaded by
path.

Loading experiments
===================

Experiments are represented as ``Experiment`` objects. Each experiment holds
one or more observables, which can be ITC heats or spectroscopic signals.

.. code-block:: python

    from linkage.experiment import Experiment

    expt = Experiment()
    expt.define_itc_observable(
        obs_file="my_itc_data.csv",
        cell_contents={"protein": 25e-6},
        syringe_contents={"ligand": 500e-6},
    )

    expt_list = [expt]

Multiple experiments covering different conditions, concentrations, or
observable types can be placed in the same list and fit simultaneously.

Building the GlobalModel
========================

``GlobalModel`` takes the experiment list and the model spec and assembles
a single callable model that predicts all observables from a shared parameter
vector.

.. code-block:: python

    from linkage import GlobalModel

    gm = GlobalModel(expt_list=expt_list,
                     model_spec="path/to/model_spec.txt")

After construction, the model exposes:

+ ``gm.param_names`` -- ordered list of all parameter names seen by the fitter.
+ ``gm.y_obs`` -- concatenated vector of all observations across experiments.
+ ``gm.y_std`` -- corresponding observation uncertainties.
+ ``gm.model_normalized(param_array)`` -- evaluates the model and returns
  normalized residuals.
+ ``gm.jacobian_normalized(param_array)`` -- returns the exact analytic
  Jacobian of the normalized residuals with respect to the parameters.
+ ``gm.hessian_normalized(param_array)`` -- returns the exact analytic
  Hessian (must be enabled at construction; see :doc:`fitters/hmc`).

Setting parameter attributes
=============================

Once the model is built, pass it to ``dataprob.setup`` to get a fitter object.
The fitter's ``param_df`` dataframe controls bounds, guesses, priors, and
which parameters are free versus fixed.

.. code-block:: python

    import dataprob

    f = dataprob.setup(gm.model_normalized,
                       method="ml",
                       vector_first_arg=True,
                       fit_parameters=gm.param_names)

    # Set a lower bound of 0 on log-K parameters
    for name in f.param_df.index:
        if name.startswith("log_K"):
            f.param_df.loc[name, "lower_bound"] = 0.0

    # Fix a nuisance parameter at a known value
    f.param_df.loc["nuisance_dil_ligand", "fixed"] = True
    f.param_df.loc["nuisance_dil_ligand", "guess"] = 0.0

Running the fit
===============

.. code-block:: python

    f.fit(y_obs=gm.y_obs, y_std=gm.y_std)

    print(f.fit_df)
    print(f.fit_quality)

Results are returned in ``f.fit_df``, a pandas DataFrame with one row per
parameter. The columns ``estimate``, ``std``, ``low_95``, and ``high_95``
summarize the fit outcome. The exact meaning of these columns depends on the
fitting method; see the individual fitter pages for details.
