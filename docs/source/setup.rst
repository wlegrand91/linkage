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

.. important::

    The symbolic parser assumes that the **last** mass balance equation in the
    ``species`` block defines the free variable — i.e. the concentration that
    will be treated as the unknown when solving the binding polynomial. In the
    Ca/EDTA example below, ``CT`` is listed last, so the parser will express
    everything as a polynomial in ``C`` and solve for it symbolically. Placing a
    different species last will change which variable is solved for, and writing
    the ``species`` block in an order that conflicts with this expectation will
    either break parsing or trigger expensive recursive solve attempts.

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

Where the coefficients come from
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The factors of 2 and 1/2 in the reparameterization rules are not arbitrary —
they come from the combinatorics of identical, independent binding sites.

When ``m`` equivalent sites all have the same microscopic affinity ``k``, the
stepwise macroscopic binding constants are related to ``k`` by:

.. math::

    K_i = \frac{m - i + 1}{i} \cdot k

The numerator counts how many empty sites are available at step ``i``; the
denominator counts how many filled sites could release a ligand. For
``m = 2`` identical sites this gives:

.. math::

    K_1 = \frac{2}{1} \cdot k = 2k \qquad K_2 = \frac{1}{2} \cdot k = \frac{k}{2}

In the six-state model the four sites are split into two high-affinity and
two low-affinity groups, each internally equivalent. The high-affinity sites
fill during steps 1 and 2, the low-affinity sites during steps 3 and 4.
Applying the formula within each group yields the four rules directly:

.. code-block:: text

    K1 = 2 * k_high    ← first of 2 high-affinity sites
    K2 = k_high / 2    ← second of 2 high-affinity sites
    K3 = 2 * k_low     ← first of 2 low-affinity sites
    K4 = k_low / 2     ← second of 2 low-affinity sites

The same logic applies to enthalpies, but because enthalpy is additive (not
multiplicative) there are no statistical factors — each step in a group of
identical sites has the same microscopic enthalpy, so the rules are simple
equalities:

.. code-block:: text

    dH_1 = dH_high
    dH_2 = dH_high
    dH_3 = dH_low
    dH_4 = dH_low

For a model with a different site arrangement — for example three equivalent
sites — the same formula applies with ``m = 3``, giving
``K1 = 3k``, ``K2 = k``, ``K3 = k/3``.

How regression parameters are determined
-----------------------------------------

When linkage parses a model spec it first determines the full set of
**physical parameters** — the quantities that appear directly in the binding
polynomial. These are derived automatically from the ``equilibria`` block:

* Every equilibrium constant named in the equilibria block (e.g. ``K1``,
  ``KE``) is a physical parameter.
* Each equilibrium constant is paired with an enthalpy parameter whose name
  is formed by replacing the leading ``K`` with ``dH_``: ``K1`` → ``dH_1``,
  ``KE`` → ``dH_E``, ``KI`` → ``dH_I``, and so on.

If there is no ``reparameterize`` block, every physical parameter is also a
**regression parameter** — the set the fitter optimises directly.

The ``reparameterize`` block modifies this set in two ways:

1. **Removes dependent parameters.** Any physical parameter that appears on
   the left-hand side of a rule is marked as dependent and is dropped from the
   regression set. In the six-state example above, ``K1``, ``K2``, ``K3``,
   ``K4``, ``dH_1``, ``dH_2``, ``dH_3``, and ``dH_4`` are all written on the
   left-hand side, so none of them appear as regression parameters.

2. **Adds new independent symbols.** Any symbol that appears on the
   right-hand side of a rule and is not already a physical parameter becomes a
   new regression parameter. In the six-state example, ``k_high``, ``k_low``,
   ``dH_high``, and ``dH_low`` are all new symbols introduced on the
   right-hand side, so they become the regression parameters the fitter sees.

The final regression set is therefore:

.. code-block:: text

    regression_params = (physical_params − dependent_params) ∪ new_RHS_symbols

Rules may be chained: a dependent parameter can appear in the expression for
another dependent parameter, and linkage will substitute iteratively until
every physical parameter is expressed purely in terms of regression parameters.
Any physical parameter not mentioned on either side of a rule passes through
unchanged — it stays in the regression set at its face value.

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
        cell_contents={"ET": 25e-6},
        syringe_contents={"CT": 500e-6},
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

    f.fit(y_obs=gm.y_obs,
          y_std=gm.y_std)

    print(f.fit_df)
    print(f.fit_quality)

Results are returned in ``f.fit_df``, a pandas DataFrame with one row per
parameter. The columns ``estimate``, ``std``, ``low_95``, and ``high_95``
summarize the fit outcome. The exact meaning of these columns depends on the
fitting method; see the individual fitter pages for details.
