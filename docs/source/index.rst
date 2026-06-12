.. linkage documentation master file

.. include:: links.rst

=======
linkage
=======

linkage is a Python library for building thermodynamic binding models from
a simple text specification and fitting them to experimental data. You define
a chemical system by writing out its equilibria, conservation laws, and any
reparameterization rules. linkage then derives the binding polynomial
symbolically using `SymPy`_, applies the implicit function theorem to produce
exact analytic Jacobians and Hessians, and exposes these objects to the
`dataprob`_ fitting framework.

The result is that gradient-based fitters receive exact derivatives rather than
finite-difference approximations. This makes regression faster, more stable,
and better suited to high-dimensional or poorly conditioned parameter spaces.

Installation
============

.. code-block:: shell

    pip install linkage

Quick example
=============

The following defines a simple 1:1 binding model and fits it to ITC data
using maximum likelihood.

.. code-block:: python

    import linkage
    import dataprob

    model_spec = """
    equilibria:
        A + B -> AB; K1

    species:
        AT = A + AB
        BT = B + AB
    """

    gm = linkage.GlobalModel(expt_list=expts, model_spec=model_spec)

    f = dataprob.setup(gm.model_normalized,
                       method="ml",
                       vector_first_arg=True,
                       fit_parameters=gm.param_names)

    f.fit(y_obs=gm.y_obs, y_std=gm.y_std)

    print(f.fit_df)

Setup and fitting
=================

.. toctree::

   setup

Maximum Likelihood
==================

.. toctree::

   fitters/ml

Bootstrap
=========

.. toctree::

   fitters/bootstrap

Bayesian MCMC (emcee)
=====================

.. toctree::

   fitters/emcee

Bayesian MCMC (PyMC)
====================

.. toctree::

   fitters/pymc

Hamiltonian Monte Carlo
=======================

.. toctree::

   fitters/hmc

Checkpointing
=============

.. toctree::

   checkpointing

Visualization
=============

.. toctree::

   visualization
