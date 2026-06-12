.. include:: ../links.rst

=====================
Bayesian MCMC (emcee)
=====================

emcee uses an affine-invariant ensemble sampler to draw samples from the
posterior distribution of model parameters. It is a gradient-free method that
works by running a collection of parallel walkers that evolve through parameter
space by proposing moves relative to each other.

For full documentation of the emcee fitter options and outputs, see the
`dataprob emcee page <dataprob-emcee_>`_.

Symbolic objects used
=====================

emcee is a gradient-free sampler. It does not use the Jacobian or the Hessian.
The ``gm.jacobian_normalized`` and ``gm.hessian_normalized`` methods provided
by linkage are not used when running this fitter.

The model is evaluated via ``gm.model_normalized`` at each proposed step.
Because emcee requires many model evaluations, the efficiency of the forward
model matters more here than the availability of exact derivatives.

Usage
=====

.. code-block:: python

    import linkage
    import dataprob

    gm = linkage.GlobalModel(expt_list=expt_list,
                             model_spec=model_spec)

    f = dataprob.setup(gm.model_normalized,
                       method="emcee",
                       vector_first_arg=True,
                       fit_parameters=gm.param_names)

    f.fit(y_obs=gm.y_obs, y_std=gm.y_std)

    print(f.fit_df)
