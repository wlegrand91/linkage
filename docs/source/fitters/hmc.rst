.. include:: ../links.rst

=======================
Hamiltonian Monte Carlo
=======================

Hamiltonian Monte Carlo (HMC) draws samples from the posterior distribution
of model parameters using leapfrog integration of Hamiltonian dynamics. It
uses gradient information to propose distant moves in parameter space that
are accepted at a high rate, making it significantly more efficient than
random-walk samplers for high-dimensional or correlated parameter spaces.

For full documentation of the HMC fitter options, outputs, and the
non-centered parameterization option, see the
`dataprob HMC page <dataprob-hmc_>`_.

Symbolic objects used
=====================

HMC is where linkage's symbolic machinery provides the greatest benefit.

**Jacobian**

The HMC fitter uses the gradient of the log posterior with respect to the
parameters at every leapfrog step. By default this gradient is computed
via finite differences. When ``gm.jacobian_normalized`` is present, dataprob
detects it automatically and uses the exact analytic Jacobian instead. This
makes each gradient evaluation exact and typically much faster than
finite-differencing, particularly for models with many parameters.

**Hessian**

The HMC fitter can use the Hessian to construct a mass matrix that
pre-conditions the sampler for the local geometry of the posterior. This
allows the sampler to take larger, better-directed steps and reduces
correlation between successive samples.

To enable the symbolic Hessian, pass ``use_symbolic_hessian=True`` when
constructing the ``GlobalModel``:

.. code-block:: python

    gm = linkage.GlobalModel(expt_list=expt_list,
                             model_spec=model_spec,
                             use_symbolic_hessian=True)

With this set, ``gm.hessian_normalized`` is available and dataprob will use
it to initialize the mass matrix before sampling begins.

.. note::

    Deriving the symbolic Hessian is computationally expensive during model
    construction, especially for models with many equilibria. For exploration
    and debugging, leave ``use_symbolic_hessian=False``. Enable it for
    production runs where sampling efficiency matters.

Usage
=====

.. code-block:: python

    import linkage
    import dataprob

    gm = linkage.GlobalModel(expt_list=expt_list,
                             model_spec=model_spec,
                             use_symbolic_hessian=True)

    f = dataprob.setup(gm.model_normalized,
                       method="hmc",
                       vector_first_arg=True,
                       fit_parameters=gm.param_names)

    f.fit(y_obs=gm.y_obs,
          y_std=gm.y_std,
          n_samples=1000,
          burn_in=500,
          output_dir="hmc_results")

    print(f.fit_df)

For long runs that may be interrupted, see :doc:`../checkpointing`.
