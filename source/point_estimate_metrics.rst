.. _point_estimates:   


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
evaluation section <bootstrapped_metrics>`. 

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

Initial Set-up
-------------------

**Step 1: Import and Initialize EquiBoots**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To begin, we instantiate the ``EquiBoots`` class with the required inputs: the 
true outcome labels (``y_test``), predicted class labels (``y_pred``), 
predicted probabilities (``y_prob``), and a DataFrame that holds sensitive
attributes like ``race`` or ``sex``.

.. note::

    ``y_pred``, ``y_prob``, ``y_test`` are defined inside the :ref:`modeling generation section <Modeling_Generation>`.


Once initialized, ``EquiBoots`` uses its internal grouping mechanism to enable 
fairness auditing by slicing the dataset into mutually exclusive subgroups based 
on each fairness variable. This slicing is a prerequisite for evaluating model
behavior across subpopulations.

The ``grouper`` method stores index-level membership for each group, ensuring 
that only groups meeting a minimum sample size are considered. This prevents 
unstable or misleading metric calculations. Once sliced, we call ``slicer`` 
to extract the ``y_true``, ``y_pred``, and ``y_prob`` values corresponding to 
each group. Finally, ``get_metrics`` is used to compute core performance metrics 
for each subgroup.

.. code:: python 

    import equiboots as eqb

    # Create fairness DataFrame
    fairness_df = X_test[['race', 'sex']].reset_index()

    eq = eqb.EquiBoots(
        y_true=y_test,
        y_prob=y_prob,
        y_pred=y_pred,
        fairness_df=fairness_df,
        fairness_vars=["race", "sex"],
    )

**Step 2: Slice Groups and Compute Point Estimates**  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the class is initialized, we slice the dataset into subgroups and compute 
performance metrics for each group. This step is critical for assessing whether 
model performance varies by group.

.. code:: python

    import equiboots as eqb

    sliced_race_data = eq.slicer("race")
    race_metrics = eq.get_metrics(sliced_race_data)

    sliced_sex_data = eq.slicer("sex")
    sex_metrics = eq.get_metrics(sliced_sex_data)


Each output is a dictionary of group names (e.g., ``'Male'``, ``'Female'``, ``'Asian'``, ``'White'``) 
mapped to performance metrics such as accuracy, AUROC, precision, or RMSE, depending on the task type.


Metrics DataFrame
-------------------------

Because these dictionaries can contain many entries and nested metric structures, 
we avoid printing them directly in documentation. Instead, we use the ``metrics_dataframe()`` 
function to transform the dictionary into a clean, filterable DataFrame.

To keep the table concise and relevant, we subset the DataFrame to include only a selected set of metrics:

- `Accuracy`
- `Precision`
- `Recall`
- `F1 Score`
- `Specificity`
- `TP Rate`
- `Prevalence`
- `Average Precision Score`
- `Calibration AUC`

.. function:: metrics_dataframe(metrics_data)

    Transforms a list of grouped metric dictionaries into a single flat DataFrame.

    :param metrics_data: A list of dictionaries, where each dictionary maps a group name to its associated performance metrics.
    :type metrics_data: List[Dict[str, Dict[str, float]]]

    :returns: A tidy DataFrame with one row per group and one column per metric. The group names are stored in the ``attribute_value`` column.
    :rtype: pd.DataFrame

.. admonition:: Notes

    - This function is typically used after computing metrics using ``eqb.get_metrics()``.
    - It flattens nested group-wise dictionaries into a readable table, enabling easy subsetting, filtering, and export.
    - Common use cases include displaying fairness-related metrics such as Accuracy, Precision, Recall, Specificity, Calibration AUC, and others across different sensitive attribute groups (e.g., race, sex).

The ``metrics_dataframe()`` function simplifies post-processing and reporting by converting the raw output of group-level metrics into a tabular format. Each row corresponds to a demographic group, and each column represents a different metric.

Below is an example of how this function is used in practice to format metrics by race:

.. code-block:: python

    race_metrics_df = eqb.metrics_dataframe(metrics_data=race_metrics)
    race_metrics_df = race_metrics_df[
        [
            "attribute_value",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Specificity",
            "TP Rate",
            "Prevalence",
            "Average Precision Score",
            "Calibration AUC",
        ]
    ]
    ## round to 3 decimal places for readability
    round(race_metrics_df, 3)

This yields a structured and readable table of group-level performance for use in reporting or further analysis.

**Output**

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    overflow:hidden;padding:5px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    font-weight:normal;overflow:hidden;padding:5px 5px;word-break:normal;}
    .tg .tg-2b7s{text-align:right;vertical-align:bottom}
    .tg .tg-bobw{font-weight:bold;text-align:center;vertical-align:bottom}
    .tg .tg-kex3{font-weight:bold;text-align:right;vertical-align:bottom}
    .tg .tg-j6zm{font-weight:bold;text-align:right;vertical-align:bottom}
    .tg .tg-8d8j{text-align:center;vertical-align:bottom}
    .tg .tg-7zrl{text-align:right;vertical-align:bottom}
    @media screen and (max-width: 767px) {.tg {width: auto !important;}.tg col {width: auto !important;}.tg-wrap {overflow-x: auto;-webkit-overflow-scrolling: touch;}}</style>
    <div class="tg-wrap"><table class="tg"><thead>
    <tr>
        <th class="tg-bobw"></th>
        <th class="tg-kex3">attribute_value</th>
        <th class="tg-j6zm">Accuracy</th>
        <th class="tg-j6zm">Precision</th>
        <th class="tg-j6zm">Recall</th>
        <th class="tg-j6zm">F1 Score</th>
        <th class="tg-j6zm">Specificity</th>
        <th class="tg-j6zm">TP Rate</th>
        <th class="tg-j6zm">Prevalence</th>
        <th class="tg-j6zm">Calibration AUC</th>
    </tr></thead>
    <tbody>
    <tr>
        <td class="tg-8d8j">0</td>
        <td class="tg-2b7s">White</td>
        <td class="tg-7zrl">0.835</td>
        <td class="tg-7zrl">0.705</td>
        <td class="tg-7zrl">0.573</td>
        <td class="tg-7zrl">0.632</td>
        <td class="tg-7zrl">0.921</td>
        <td class="tg-7zrl">0.573</td>
        <td class="tg-7zrl">0.247</td>
        <td class="tg-7zrl">0.092</td>
    </tr>
    <tr>
        <td class="tg-8d8j">1</td>
        <td class="tg-2b7s">White</td>
        <td class="tg-7zrl">0.885</td>
        <td class="tg-7zrl">0.815</td>
        <td class="tg-7zrl">0.657</td>
        <td class="tg-7zrl">0.727</td>
        <td class="tg-7zrl">0.955</td>
        <td class="tg-7zrl">0.657</td>
        <td class="tg-7zrl">0.233</td>
        <td class="tg-7zrl">0.029</td>
    </tr>
    <tr>
        <td class="tg-8d8j">2</td>
        <td class="tg-2b7s">White</td>
        <td class="tg-7zrl">0.877</td>
        <td class="tg-7zrl">0.743</td>
        <td class="tg-7zrl">0.681</td>
        <td class="tg-7zrl">0.710</td>
        <td class="tg-7zrl">0.933</td>
        <td class="tg-7zrl">0.681</td>
        <td class="tg-7zrl">0.222</td>
        <td class="tg-7zrl">0.055</td>
    </tr>
    <tr>
        <td class="tg-8d8j">3</td>
        <td class="tg-2b7s">White</td>
        <td class="tg-7zrl">0.862</td>
        <td class="tg-7zrl">0.804</td>
        <td class="tg-7zrl">0.675</td>
        <td class="tg-7zrl">0.734</td>
        <td class="tg-7zrl">0.935</td>
        <td class="tg-7zrl">0.675</td>
        <td class="tg-7zrl">0.282</td>
        <td class="tg-7zrl">0.033</td>
    </tr>
    <tr>
        <td class="tg-8d8j">4</td>
        <td class="tg-2b7s">White</td>
        <td class="tg-7zrl">0.868</td>
        <td class="tg-7zrl">0.744</td>
        <td class="tg-7zrl">0.709</td>
        <td class="tg-7zrl">0.726</td>
        <td class="tg-7zrl">0.920</td>
        <td class="tg-7zrl">0.709</td>
        <td class="tg-7zrl">0.247</td>
        <td class="tg-7zrl">0.061</td>
    </tr>
    <tr>
        <td class="tg-8d8j">...</td>
        <td class="tg-2b7s">...</td>
        <td class="tg-7zrl">...</td>
        <td class="tg-7zrl">...</td>
        <td class="tg-7zrl">...</td>
        <td class="tg-7zrl">...</td>
        <td class="tg-7zrl">...</td>
        <td class="tg-7zrl">...</td>
        <td class="tg-7zrl">...</td>
        <td class="tg-7zrl">...</td>
    </tr>
    <tr>
        <td class="tg-8d8j">14998</td>
        <td class="tg-2b7s">Asian-Pac-Islander</td>
        <td class="tg-7zrl">0.808</td>
        <td class="tg-7zrl">0.600</td>
        <td class="tg-7zrl">0.500</td>
        <td class="tg-7zrl">0.545</td>
        <td class="tg-7zrl">0.900</td>
        <td class="tg-7zrl">0.500</td>
        <td class="tg-7zrl">0.231</td>
        <td class="tg-7zrl">0.281</td>
    </tr>
    <tr>
        <td class="tg-8d8j">14999</td>
        <td class="tg-2b7s">Asian-Pac-Islander</td>
        <td class="tg-7zrl">0.923</td>
        <td class="tg-7zrl">1.000</td>
        <td class="tg-7zrl">0.600</td>
        <td class="tg-7zrl">0.750</td>
        <td class="tg-7zrl">1.000</td>
        <td class="tg-7zrl">0.600</td>
        <td class="tg-7zrl">0.192</td>
        <td class="tg-7zrl">0.304</td>
    </tr>
    <tr>
        <td class="tg-8d8j">15000</td>
        <td class="tg-2b7s">Asian-Pac-Islander</td>
        <td class="tg-7zrl">0.808</td>
        <td class="tg-7zrl">0.750</td>
        <td class="tg-7zrl">0.429</td>
        <td class="tg-7zrl">0.545</td>
        <td class="tg-7zrl">0.947</td>
        <td class="tg-7zrl">0.429</td>
        <td class="tg-7zrl">0.269</td>
        <td class="tg-7zrl">0.232</td>
    </tr>
    <tr>
        <td class="tg-8d8j">15001</td>
        <td class="tg-2b7s">Asian-Pac-Islander</td>
        <td class="tg-7zrl">0.808</td>
        <td class="tg-7zrl">0.625</td>
        <td class="tg-7zrl">0.714</td>
        <td class="tg-7zrl">0.667</td>
        <td class="tg-7zrl">0.842</td>
        <td class="tg-7zrl">0.714</td>
        <td class="tg-7zrl">0.269</td>
        <td class="tg-7zrl">0.175</td>
    </tr>
    <tr>
        <td class="tg-8d8j">15002</td>
        <td class="tg-2b7s">Asian-Pac-Islander</td>
        <td class="tg-7zrl">0.731</td>
        <td class="tg-7zrl">0.571</td>
        <td class="tg-7zrl">0.500</td>
        <td class="tg-7zrl">0.533</td>
        <td class="tg-7zrl">0.833</td>
        <td class="tg-7zrl">0.500</td>
        <td class="tg-7zrl">0.308</td>
        <td class="tg-7zrl">0.362</td>
    </tr>
    </tbody></table></div>

.. raw:: html

    <div style="height: 40px;"></div>





.. [1] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.