.. include:: ../links.rst

====================
Bayesian MCMC (PyMC)
====================

PyMC uses the NUTS (No-U-Turn Sampler) to draw samples from the posterior
distribution of model parameters. It supports multiple chains and provides
convergence diagnostics. NUTS is a gradient-based sampler that adapts its
step size and trajectory length automatically.

For full documentation of the PyMC fitter options and outputs, see the
`dataprob PyMC page <dataprob-pymc_>`_.

Symbolic objects used
=====================

PyMC computes gradients internally via its own automatic differentiation
backend (Aesara/PyTensor). It does not use ``gm.jacobian_normalized`` or
``gm.hessian_normalized`` from linkage.

The model is evaluated through a PyMC-compatible wrapper around
``gm.model_normalized``. As with emcee, the efficiency of the forward model
is the main performance lever when using this fitter.

Usage
=====

.. code-block:: python

    import linkage
    import dataprob

    gm = linkage.GlobalModel(expt_list=expt_list,
                             model_spec=model_spec)

    f = dataprob.setup(gm.model_normalized,
                       method="pymc",
                       vector_first_arg=True,
                       fit_parameters=gm.param_names)

    f.fit(y_obs=gm.y_obs, y_std=gm.y_std)

    print(f.fit_df)
