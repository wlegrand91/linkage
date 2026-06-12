.. include:: links.rst

=============
Checkpointing
=============

HMC sampling runs can take a long time for complex models. The HMC fitter in
dataprob supports checkpointing: it periodically saves the sampler state to
disk so that an interrupted run can be resumed from where it left off rather
than restarted from scratch.

Saving checkpoints
==================

Pass an ``output_dir`` argument to ``f.fit()`` to enable checkpointing. The
fitter will create the directory if it does not exist and write a
``hmc_checkpoint.npz`` file at regular intervals.

.. code-block:: python

    f.fit(y_obs=gm.y_obs,
          y_std=gm.y_std,
          n_samples=2000,
          burn_in=500,
          output_dir="hmc_results",
          checkpoint_steps=100)

``checkpoint_steps`` controls how often (in samples) the checkpoint file is
written. The default is 0, which disables intermediate checkpoints and only
writes a final checkpoint when sampling completes.

Resuming from a checkpoint
==========================

To resume a run, pass the same ``output_dir`` (or the path to the
``hmc_checkpoint.npz`` file directly) as the ``resume_from`` argument.
Burn-in is skipped and sampling continues from the saved state.

.. code-block:: python

    f.fit(y_obs=gm.y_obs,
          y_std=gm.y_std,
          n_samples=2000,
          burn_in=500,
          output_dir="hmc_results",
          resume_from="hmc_results")

The fitter will load the saved samples, step size, and random state from the
checkpoint and append new samples until the total requested number is reached.

.. note::

    The ``n_samples`` argument still refers to the total number of samples
    requested, not additional samples on top of the checkpoint. If the
    checkpoint already contains 1500 samples and ``n_samples=2000``, the
    resumed run will draw 500 more samples.

What is saved
=============

The ``hmc_checkpoint.npz`` file contains:

+ All samples collected so far (post-burn-in).
+ The current position of the sampler in parameter space.
+ The current adapted step size.
+ The acceptance count.
+ The random number generator state, so the resumed run is reproducible.

A ``fit_summary.csv`` file is also written to ``output_dir`` after sampling
completes (or is interrupted). It records the HMC configuration and diagnostics
such as acceptance rate, step size, and total run time.
