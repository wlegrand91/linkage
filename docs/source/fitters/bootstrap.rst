.. include:: ../links.rst

=========
Bootstrap
=========

Bootstrap resampling estimates the distribution of parameter values consistent
with the observed data by repeatedly sampling from the observation uncertainty,
refitting the model to each pseudo-replicate dataset, and collecting the
resulting parameter estimates. It requires no assumptions about the shape of
the parameter distribution.

For full documentation of the bootstrap fitter options and outputs, see the
`dataprob Bootstrap page <dataprob-bootstrap_>`_.

Symbolic objects used
=====================

The bootstrap fitter runs many independent ML fits, one per pseudo-replicate.
Each of those fits uses the **Jacobian** if available, just as the ML fitter
does. linkage provides ``gm.jacobian_normalized``, so the analytic Jacobian is
used for every bootstrap replicate automatically.

The Hessian is not used by the bootstrap fitter.

Usage
=====

.. code-block:: python

    import linkage
    import dataprob

    gm = linkage.GlobalModel(expt_list=expt_list,
                             model_spec=model_spec)

    f = dataprob.setup(gm.model_normalized,
                       method="bootstrap",
                       vector_first_arg=True,
                       fit_parameters=gm.param_names)

    f.fit(y_obs=gm.y_obs,
          y_std=gm.y_std,
          num_bootstrap=500)

    print(f.fit_df)
