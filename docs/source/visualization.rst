.. include:: links.rst

=============
Visualization
=============

All visualization functions are provided by `dataprob`_. They take a fitted
fitter object ``f`` (for which ``f.fit()`` has been run) and return a
matplotlib figure.

linkage does not currently provide its own plotting functions.

plot_summary
============

``dataprob.plot_summary`` produces a four-panel diagnostic plot showing the
fit quality.

.. code-block:: python

    import dataprob

    fig = dataprob.plot_summary(f)
    fig.savefig("summary.pdf")

The central panel shows the observed data with error bars and the model
evaluated at the best-fit parameters (red line), along with a cloud of gray
lines showing the model evaluated at parameter samples. The lower and right
panels show weighted residuals. The bottom-right panel shows a histogram of
the residuals.

plot_corner
===========

``dataprob.plot_corner`` shows pairwise parameter distributions, useful for
assessing parameter uncertainty and covariance.

.. code-block:: python

    fig = dataprob.plot_corner(f)

For models with nuisance parameters, use ``filter_params`` to focus on the
parameters of interest:

.. code-block:: python

    fig = dataprob.plot_corner(f, filter_params=["nuisance"])

This removes any parameter whose name contains the string ``"nuisance"``.

plot_fit
========

``dataprob.plot_fit`` plots the observed data and the model line alone,
without the residual panels.

.. code-block:: python

    fig = dataprob.plot_fit(f)

plot_residuals
==============

``dataprob.plot_residuals`` plots the weighted residuals as a function of
observation index.

.. code-block:: python

    fig = dataprob.plot_residuals(f)

plot_residuals_hist
===================

``dataprob.plot_residuals_hist`` plots a histogram of the weighted residuals.

.. code-block:: python

    fig = dataprob.plot_residuals_hist(f)
