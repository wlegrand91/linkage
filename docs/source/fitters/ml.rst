.. include:: ../links.rst

==================
Maximum Likelihood
==================

Maximum likelihood (ML) fitting finds the parameter vector that minimizes the
weighted sum of squared residuals between the model and observations. It is the
fastest method and a good starting point for any analysis.

For full documentation of the ML fitter options and outputs, see the
`dataprob ML page <dataprob-ml_>`_.

Symbolic objects used
=====================

The ML fitter in dataprob uses the **Jacobian** when one is available. linkage
provides ``gm.jacobian_normalized``, which returns the exact analytic Jacobian
of the normalized residuals with respect to all free parameters. dataprob
detects this method automatically via duck typing (checking for a
``jacobian_normalized`` attribute on the model object) and passes it to the
underlying optimizer.

This replaces the default finite-difference Jacobian approximation with an
exact derivative, which improves convergence speed and accuracy, particularly
for stiff or ill-conditioned parameter spaces.

The Hessian is not used by the ML fitter.

Usage
=====

.. code-block:: python

    import linkage
    import dataprob

    gm = linkage.GlobalModel(expt_list=expt_list,
                             model_spec=model_spec)

    f = dataprob.setup(gm.model_normalized,
                       method="ml",
                       vector_first_arg=True,
                       fit_parameters=gm.param_names)

    f.fit(y_obs=gm.y_obs,
          y_std=gm.y_std)

    print(f.fit_df)
