.. _point_estimates:   

.. _target-link:


Point Estimate Evaluation
==========================================

After training a model and preparing predictions, EquiBoots can be used to 
evaluate how your model performs across different demographic groups. The most 
basic step in this process is calculating point estimates. These are performance 
metrics for each group without resampling or bootstrapping.

EquiBoots supports the computation of group-specific and overall point estimates 
for performance metrics across classification and regression tasks. These estimates 
form the basis for fairness auditing by revealing how models perform across 
different subpopulations or sensitive attributes.

This section demonstrates how to compute group-wise performance metrics using 
model outputs and fairness variables from the Adult Income dataset [1]_. For 
bootstrapped confidence intervals, refer to the :ref:`bootstrapped metrics 
evaluation section <Bootstrapped_Metrics>`. 

Supported Metrics
-------------------------


For classification tasks, the following metrics are supported:

- Accuracy, Precision, Recall, F1-score

- AUROC, AUPRC (for probabilistic models)

- Calibration Area Under The Curve 

- Log Loss, Brier Score

For regression tasks:

- :math:`R^2, MAE, MSE, RMSE`

- Group-based residual plots

Implementation
-------------------

Step 1: Import and Initialize EquiBoots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We begin by initializing the ``EquiBoots`` class using the predicted labels, 
predicted probabilities, true labels, and the DataFrame containing sensitive 
attributes (e.g., sex, race, education).

.. note::

    ``y_pred``, ``y_prob``, ``y_test`` are defined inside the :ref:`modeling generation section <Modeling_Generation>`.

Point estimates are computed using the ``EquiBoots`` class and its ``grouper`` method:

.. code:: python 

    import equiboots as eqb

    eq = EquiBoots(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        fairness_df=fairness_df,
        fairness_vars=["sex", "race"],
        task="binary_classification"
    )



.. [1] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.