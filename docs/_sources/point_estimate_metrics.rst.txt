.. _point_estimates:   

.. raw:: html

   <div class="no-click">

.. image:: ../assets/EquiBoots.png
   :alt: EquiBoots Logo
   :align: left
   :width: 300px

.. raw:: html
   
   <div style="height: 130px;"></div></div>



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
evaluation section <bootstrapped_estimates>`. 

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

.. _note:

    - This function is used after computing metrics using ``eqb.get_metrics()``.
    - It flattens nested group-wise dictionaries into a readable table, enabling easy subsetting, filtering, and export.
    - Common use cases include displaying fairness-related metrics such as Accuracy, Precision, Recall, Specificity, Calibration AUC, and others across different sensitive attribute groups (e.g., race, sex).

The ``metrics_dataframe()`` function simplifies post-processing and reporting by converting the raw output of group-level metrics into a tabular format. Each row corresponds to a demographic group, and each column represents a different metric.

Below is an example of how this function is used in practice to format metrics by race:

.. code-block:: python

    import equiboots as eqb

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
    overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-2b7s{text-align:right;vertical-align:bottom}
    .tg .tg-8d8j{text-align:center;vertical-align:bottom}
    .tg .tg-kex3{font-weight:bold;text-align:right;vertical-align:bottom}
    @media screen and (max-width: 767px) {.tg {width: auto !important;}.tg col {width: auto !important;}.tg-wrap {overflow-x: auto;-webkit-overflow-scrolling: touch;}}</style>
    <div class="tg-wrap"><table class="tg"><thead>
    <tr>
        <th class="tg-8d8j"></th>
        <th class="tg-kex3">attribute_value</th>
        <th class="tg-kex3">Accuracy</th>
        <th class="tg-kex3">Precision</th>
        <th class="tg-kex3">Recall</th>
        <th class="tg-kex3">F1 Score</th>
        <th class="tg-kex3">Specificity</th>
        <th class="tg-kex3">TP Rate</th>
        <th class="tg-kex3">Prevalence</th>
        <th class="tg-kex3">Calibration AUC</th>
    </tr></thead>
    <tbody>
    <tr>
        <td class="tg-8d8j">0</td>
        <td class="tg-2b7s">White</td>
        <td class="tg-2b7s">0.853</td>
        <td class="tg-2b7s">0.761</td>
        <td class="tg-2b7s">0.638</td>
        <td class="tg-2b7s">0.694</td>
        <td class="tg-2b7s">0.929</td>
        <td class="tg-2b7s">0.638</td>
        <td class="tg-2b7s">0.262</td>
        <td class="tg-2b7s">0.040</td>
    </tr>
    <tr>
        <td class="tg-8d8j">1</td>
        <td class="tg-2b7s">Black</td>
        <td class="tg-2b7s">0.931</td>
        <td class="tg-2b7s">0.861</td>
        <td class="tg-2b7s">0.549</td>
        <td class="tg-2b7s">0.670</td>
        <td class="tg-2b7s">0.987</td>
        <td class="tg-2b7s">0.549</td>
        <td class="tg-2b7s">0.128</td>
        <td class="tg-2b7s">0.054</td>
    </tr>
    <tr>
        <td class="tg-8d8j">2</td>
        <td class="tg-2b7s">Asian-Pac-Islander</td>
        <td class="tg-2b7s">0.826</td>
        <td class="tg-2b7s">0.760</td>
        <td class="tg-2b7s">0.543</td>
        <td class="tg-2b7s">0.633</td>
        <td class="tg-2b7s">0.934</td>
        <td class="tg-2b7s">0.543</td>
        <td class="tg-2b7s">0.277</td>
        <td class="tg-2b7s">0.140</td>
    </tr>
    <tr>
        <td class="tg-8d8j">3</td>
        <td class="tg-2b7s">Amer-Indian-Eskimo</td>
        <td class="tg-2b7s">0.879</td>
        <td class="tg-2b7s">0.444</td>
        <td class="tg-2b7s">0.364</td>
        <td class="tg-2b7s">0.400</td>
        <td class="tg-2b7s">0.943</td>
        <td class="tg-2b7s">0.364</td>
        <td class="tg-2b7s">0.111</td>
        <td class="tg-2b7s">0.323</td>
    </tr>
    <tr>
        <td class="tg-8d8j">4</td>
        <td class="tg-2b7s">Other</td>
        <td class="tg-2b7s">0.958</td>
        <td class="tg-2b7s">1.000</td>
        <td class="tg-2b7s">0.500</td>
        <td class="tg-2b7s">0.667</td>
        <td class="tg-2b7s">1.000</td>
        <td class="tg-2b7s">0.500</td>
        <td class="tg-2b7s">0.083</td>
        <td class="tg-2b7s">0.277</td>
    </tr>
    </tbody></table></div>

.. raw:: html

    <div style="height: 40px;"></div>

Statistical Tests
------------------------

After computing point estimates for different demographic groups, we may want to 
assess whether observed differences in model performance are statistically significant. 
This is particularly important when determining if disparities are due to random 
variation or reflect systematic bias.

EquiBoots provides a method to conduct hypothesis testing across group-level metrics. 
The ``analyze_statistical_significance`` function performs appropriate statistical 
tests—such as Chi-square tests for classification tasks—while supporting multiple 
comparison adjustments.

.. function:: analyze_statistical_significance(metric_dict, var_name, test_config, differences=None)

    **Performs statistical significance testing of metric differences between groups.**

    This method compares model performance across subgroups (e.g., race, sex) to determine whether the differences in metrics (e.g., accuracy, F1 score) are statistically significant. It supports multiple test types and adjustment methods for robust group-level comparison.

    :param metric_dict: Dictionary of metrics returned by ``get_metrics()``, where each key is a group name and values are metric dictionaries.
    :type metric_dict: dict

    :param var_name: The name of the sensitive attribute or grouping variable (e.g., ``"race"``, ``"sex"``).
    :type var_name: str

    :param test_config: Configuration dictionary defining how the statistical test is performed. The following keys are supported:

        - ``test_type``: Type of test to use (e.g., ``"chi_square"``, ``"bootstrap"``).
        - ``alpha``: Significance threshold (default: 0.05).
        - ``adjust_method``: Correction method for multiple comparisons (e.g., ``"bonferroni"``, ``"fdr_bh"``, ``"holm"``, or ``"none"``).
        - ``confidence_level``: Confidence level used to compute intervals (e.g., ``0.95``).
        - ``classification_task``: Specify if the model task is ``"binary_classification"`` or ``"multiclass_classification"``.
    
    :type test_config: dict

    :param differences: Optional precomputed list of raw metric differences (default is ``None``; typically not required).
    :type differences: list, optional

    :returns: A nested dictionary containing statistical test results for each metric, with each value being a ``StatTestResult`` object that includes:
        
        - test statistic
        - raw and adjusted p-values
        - confidence intervals
        - significance flags (``True`` / ``False``)
        - effect sizes (e.g., Cohen’s d, rank-biserial correlation)

    :rtype: Dict[str, Dict[str, StatTestResult]]

    :raises ValueError: If ``test_config`` is not provided or is ``None``.


This function returns a dictionary where each key is a metric name and the 
corresponding value is another dictionary mapping each group to its ``StatTestResult``.

Example
~~~~~~~~~~~

The following example demonstrates how to configure and run these tests on 
performance metrics for the ``race`` and ``sex`` subgroups:

.. code:: python

    test_config = {
        "test_type": "chi_square",
        "alpha": 0.05,
        "adjust_method": "bonferroni",
        "confidence_level": 0.95,
        "classification_task": "binary_classification",
    }
    stat_test_results_race = eq.analyze_statistical_significance(
        race_metrics, "race", test_config
    )

    stat_test_results_sex = eq.analyze_statistical_significance(
        sex_metrics, "sex", test_config
    )

    overall_stat_results = {
        "sex": stat_test_results_sex,
        "race": stat_test_results_race,
    }


Statistical Significance Plots
--------------------------------

EquiBoots supports formal statistical testing to assess whether differences in 
performance metrics across demographic groups are statistically significant.

When auditing models for fairness, it’s important not just to observe differences 
in metrics like accuracy or recall, but to determine whether these differences are 
**statistically significant**. EquiBoots provides built-in support for this analysis 
via omnibus and pairwise statistical tests.

Test Setup
~~~~~~~~~~~

- EquiBoots uses **chi-square tests** to evaluate:
  
  - Whether overall performance disparities across groups are significant (omnibus test).
  - If so, which specific groups significantly differ from the reference (pairwise tests).

- Reference groups for each fairness variable can be set manually during class initialization using the ``reference_groups`` parameter:

  .. code:: python

      eq = eqb.EquiBoots(
          y_true=...,
          y_pred=...,
          y_prob=...,
          fairness_df=...,
          fairness_vars=["race", "sex"],
          reference_groups=["white", "female"]
      )

Group Metrics Point Plot
================================

.. function:: eq_group_metrics_point_plot(group_metrics, metric_cols, category_names, include_legend=True, cmap='tab20c', save_path=None, filename='Point_Disparity_Metrics', strict_layout=True, figsize=None, show_grid=True, plot_thresholds=(0.0, 2.0), show_pass_fail=False, y_lim=None, leg_cols=3, raw_metrics=False, statistical_tests=None, show_reference=True, **plot_kwargs)

   Creates a grid of point plots for visualizing metric values (or disparities) across sensitive groups and multiple categories (e.g., race, sex). Each subplot corresponds to one (metric, category) combination, and groups are colored or flagged based on significance or pass/fail criteria.

   :param group_metrics: A list of dictionaries where each dictionary maps group names to their respective metric values for one category.
   :type group_metrics: list[dict[str, dict[str, float]]]

   :param metric_cols: List of metric names to plot (one per row).
   :type metric_cols: list[str]

   :param category_names: Names of each category corresponding to group_metrics (one per column).
   :type category_names: list[str]

   :param include_legend: Whether to display the legend on the plot.
   :type include_legend: bool

   :param cmap: Colormap used to distinguish groups.
   :type cmap: str

   :param save_path: Directory path where the plot should be saved. If None, the plot is shown.
   :type save_path: str or None

   :param filename: Filename for saving the plot (without extension).
   :type filename: str

   :param strict_layout: Whether to apply tight layout spacing.
   :type strict_layout: bool

   :param figsize: Tuple for figure size (width, height).
   :type figsize: tuple[float, float] or None

   :param show_grid: Toggle for showing gridlines on plots.
   :type show_grid: bool

   :param plot_thresholds: A tuple (lower, upper) for pass/fail thresholds.
   :type plot_thresholds: tuple[float, float]

   :param show_pass_fail: Whether to color points based on pass/fail evaluation rather than group color.
   :type show_pass_fail: bool

   :param y_lim: Y-axis limits as a (min, max) tuple.
   :type y_lim: tuple[float, float] or None

   :param leg_cols: Number of columns in the group legend.
   :type leg_cols: int

   :param raw_metrics: Whether the input metrics are raw values (True) or already calculated disparities (False).
   :type raw_metrics: bool

   :param statistical_tests: Dictionary mapping categories to their statistical test results, used for annotating groups with significance markers.
   :type statistical_tests: dict or None

   :param show_reference: Whether to plot the horizontal reference line (e.g., y=1 for ratios).
   :type show_reference: bool

   :param plot_kwargs: Additional keyword arguments passed to `sns.scatterplot`.
   :type plot_kwargs: dict[str, Union[str, float]]



Once tests are computed, the ``eq_group_metrics_point_plot`` function can 
visualize point estimates along with statistical significance annotations:

.. code:: python

    eqb.eq_group_metrics_point_plot(
        group_metrics=[race_metrics, sex_metrics],
        metric_cols=[
            "Accuracy",
            "Precision",
            "Recall",
        ],
        category_names=["race", "sex"],
        figsize=(6, 8),
        include_legend=True,
        raw_metrics=True,
        show_grid=True,
        y_lim=(0, 1.1),
        statistical_tests=overall_stat_results,
        show_pass_fail=False,
        show_reference=False,
        y_lims = {(0,0): (0.70, 1.0), (0,1): (0.70, 1.0)}
    )

**Output**

.. raw:: html

   <div class="no-click">

.. image:: ../assets/stats_based_point_estimates_plot.png
   :alt: Statistically-Based Point Estimate Plot
   :align: center
   :width: 550px

.. raw:: html

    <div style="height: 40px;"></div></div>


The chart above summarizes how model performance varies across race and sex groups 
for three key metrics: **Accuracy**, **Precision**, and **Recall**.

Each **subplot** corresponds to a single metric, plotted separately for race (left) and sex (right).

Here's how to read the plot:

- Each **point** shows the average metric score for a demographic group.
- **Letters (A–G)** label the groups (e.g., A = Amer-Indian-Eskimo, B = Asian-Pac-Islander), with the full mapping provided in the legend.
- The **star symbol (★)** below a group axis label indicates that the **omnibus test** for the corresponding fairness attribute (e.g., race or sex) was statistically significant overall.
- The **triangle symbol (▲)** denotes groups that differ **significantly from the reference group**, as determined by pairwise statistical tests (e.g., Bonferroni-adjusted chi-square).
- Color-coding helps distinguish categories and corresponds to the legend at the top.

This visualization reveals whether disparities exist not only **numerically**, but also **statistically**, helping validate whether observed group-level differences are likely due to bias or simply random variation.


Statistical Metrics table
-----------------------------


Once statistical tests have been performed, we can summarize the results in a structured table that shows:

- The **performance metrics** for each group.
- Whether the **omnibus test** detected any significant overall differences.
- Which **individual groups** differ significantly from the reference group.

This is done using the ``metrics_table`` function from EquiBoots, which takes in group metrics, test results, and the name of the reference group:

.. function:: metrics_table(metrics, statistical_tests=None, differences=None, reference_group=None)

    :param metrics: A dictionary or list of dictionaries containing metric results per group. This can either be point estimate output from ``get_metrics`` or bootstrapped results.
    :type metrics: dict or list

    :param statistical_tests: Output from ``analyze_statistical_significance`` containing omnibus and pairwise test results. If provided, annotations will be added to the output table to reflect significance.
    :type statistical_tests: dict, optional

    :param differences: A list of bootstrapped difference dictionaries returned from ``calculate_differences``. If provided, the function will average these differences and annotate the results if significant.
    :type differences: list of dict, optional

    :param reference_group: Name of the reference group used in pairwise comparisons. Only needed if displaying pairwise significance for bootstrapped differences.
    :type reference_group: str, optional

    :returns: A pandas DataFrame where rows are metric names and columns are group names. If ``statistical_tests`` is provided:
        - Omnibus test significance is marked with an asterisk (``*``) next to column names.
        - Pairwise group significance (vs. reference) is marked with a triangle (``▲``).
    :rtype: pd.DataFrame
    
.. note::

    - The function supports **both point estimates and bootstrapped results**.
    - When using bootstrapped differences, it computes the **mean difference** for each metric across iterations.
    - Automatically drops less commonly visualized metrics like Brier Score, Log Loss, and Prevalence for clarity if significance annotations are active.


.. code:: python

    from equiboots.tables import metrics_table

    stat_metrics_table_point = metrics_table(
        race_metrics,
        statistical_tests=stat_test_results_race,
        reference_group="White",
    )

You can then display the table as follows:

.. code:: python

    ## Table with metrics per group and statistical significance shown on 
    ## columns for omnibus and/or pairwise

    stat_metrics_table_point

The resulting table displays one row per group and one column per metric. Symbols like ``*`` and ``▲`` appear in the appropriate cells to indicate significance:

- ★ marks metrics where the **omnibus test** found significant variation across all groups.
- ▲ marks metrics where a specific group differs significantly from the **reference group**.

This format provides a concise, interpretable snapshot of where disparities are statistically supported in your model outputs.

.. raw:: html

    <style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;}
    .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
    font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
    .tg .tg-2b7s{text-align:right;vertical-align:bottom}
    .tg .tg-7zrl{text-align:left;vertical-align:bottom}
    .tg .tg-kex3{font-weight:bold;text-align:right;vertical-align:bottom}
    @media screen and (max-width: 767px) {.tg {width: auto !important;}.tg col {width: auto !important;}.tg-wrap {overflow-x: auto;-webkit-overflow-scrolling: touch;}}</style>
    <div class="tg-wrap"><table class="tg"><thead>
    <tr>
        <th class="tg-7zrl"></th>
        <th class="tg-kex3">White *</th>
        <th class="tg-kex3">Black * ▲</th>
        <th class="tg-kex3">Asian-Pac-Islander *</th>
        <th class="tg-kex3">Amer-Indian-Eskimo * ▲</th>
        <th class="tg-kex3">Other * ▲</th>
    </tr></thead>
    <tbody>
    <tr>
        <td class="tg-7zrl">Accuracy</td>
        <td class="tg-2b7s">0.853</td>
        <td class="tg-2b7s">0.931</td>
        <td class="tg-2b7s">0.826</td>
        <td class="tg-2b7s">0.879</td>
        <td class="tg-2b7s">0.958</td>
    </tr>
    <tr>
        <td class="tg-7zrl">Precision</td>
        <td class="tg-2b7s">0.761</td>
        <td class="tg-2b7s">0.861</td>
        <td class="tg-2b7s">0.76</td>
        <td class="tg-2b7s">0.444</td>
        <td class="tg-2b7s">1</td>
    </tr>
    <tr>
        <td class="tg-7zrl">Recall</td>
        <td class="tg-2b7s">0.638</td>
        <td class="tg-2b7s">0.549</td>
        <td class="tg-2b7s">0.543</td>
        <td class="tg-2b7s">0.364</td>
        <td class="tg-2b7s">0.5</td>
    </tr>
    <tr>
        <td class="tg-7zrl">F1 Score</td>
        <td class="tg-2b7s">0.694</td>
        <td class="tg-2b7s">0.67</td>
        <td class="tg-2b7s">0.633</td>
        <td class="tg-2b7s">0.4</td>
        <td class="tg-2b7s">0.667</td>
    </tr>
    <tr>
        <td class="tg-7zrl">Specificity</td>
        <td class="tg-2b7s">0.929</td>
        <td class="tg-2b7s">0.987</td>
        <td class="tg-2b7s">0.934</td>
        <td class="tg-2b7s">0.943</td>
        <td class="tg-2b7s">1</td>
    </tr>
    <tr>
        <td class="tg-7zrl">TP Rate</td>
        <td class="tg-2b7s">0.638</td>
        <td class="tg-2b7s">0.549</td>
        <td class="tg-2b7s">0.543</td>
        <td class="tg-2b7s">0.364</td>
        <td class="tg-2b7s">0.5</td>
    </tr>
    <tr>
        <td class="tg-7zrl">FP Rate</td>
        <td class="tg-2b7s">0.071</td>
        <td class="tg-2b7s">0.013</td>
        <td class="tg-2b7s">0.066</td>
        <td class="tg-2b7s">0.057</td>
        <td class="tg-2b7s">0</td>
    </tr>
    <tr>
        <td class="tg-7zrl">FN Rate</td>
        <td class="tg-2b7s">0.362</td>
        <td class="tg-2b7s">0.451</td>
        <td class="tg-2b7s">0.457</td>
        <td class="tg-2b7s">0.636</td>
        <td class="tg-2b7s">0.5</td>
    </tr>
    <tr>
        <td class="tg-7zrl">TN Rate</td>
        <td class="tg-2b7s">0.929</td>
        <td class="tg-2b7s">0.987</td>
        <td class="tg-2b7s">0.934</td>
        <td class="tg-2b7s">0.943</td>
        <td class="tg-2b7s">1</td>
    </tr>
    <tr>
        <td class="tg-7zrl">TP</td>
        <td class="tg-2b7s">1375</td>
        <td class="tg-2b7s">62</td>
        <td class="tg-2b7s">38</td>
        <td class="tg-2b7s">4</td>
        <td class="tg-2b7s">3</td>
    </tr>
    <tr>
        <td class="tg-7zrl">FP</td>
        <td class="tg-2b7s">432</td>
        <td class="tg-2b7s">10</td>
        <td class="tg-2b7s">12</td>
        <td class="tg-2b7s">5</td>
        <td class="tg-2b7s">0</td>
    </tr>
    <tr>
        <td class="tg-7zrl">FN</td>
        <td class="tg-2b7s">780</td>
        <td class="tg-2b7s">51</td>
        <td class="tg-2b7s">32</td>
        <td class="tg-2b7s">7</td>
        <td class="tg-2b7s">3</td>
    </tr>
    <tr>
        <td class="tg-7zrl">TN</td>
        <td class="tg-2b7s">5631</td>
        <td class="tg-2b7s">760</td>
        <td class="tg-2b7s">171</td>
        <td class="tg-2b7s">83</td>
        <td class="tg-2b7s">66</td>
    </tr>
    <tr>
        <td class="tg-7zrl">Predicted Prevalence</td>
        <td class="tg-2b7s">0.22</td>
        <td class="tg-2b7s">0.082</td>
        <td class="tg-2b7s">0.198</td>
        <td class="tg-2b7s">0.091</td>
        <td class="tg-2b7s">0.042</td>
    </tr>
    </tbody></table></div>

.. raw:: html

    <div style="height: 40px;"></div>


Group Curve Plots
==================

To help visualize how model performance varies across sensitive groups, EquiBoots 
provides a convenient plotting function for generating ROC, Precision-Recall, and 
Calibration curves by subgroup. These visualizations are essential for identifying 
disparities in predictive behavior and diagnosing potential fairness issues.

The function below allows you to create either overlaid or per-group subplots, 
customize curve aesthetics, exclude small or irrelevant groups, and optionally save plots for reporting.

After slicing your data using the ``slicer()`` method and organizing group-specific 
``y_true`` and ``y_prob`` values, you can pass the resulting dictionary to 
``eq_plot_group_curves`` to generate interpretable, publication-ready visuals.

.. function:: eq_plot_group_curves(data, curve_type="roc", n_bins=10, decimal_places=2, curve_kwgs=None, line_kwgs=None, title="Curve by Group", filename="group", save_path=None, figsize=(8, 6), dpi=100, subplots=False, n_cols=2, n_rows=None, group=None, color_by_group=True, exclude_groups=0, show_grid=True, lowess=0, lowess_kwargs=None, shade_area=False, plot_hist=False)

    Plots ROC, Precision-Recall, or Calibration curves by demographic group. Supports overlaid and subplot layouts, optional smoothing with LOWESS, shaded areas, and histogram overlays for calibration curves.

    :param data: Dictionary mapping group names to dictionaries containing ``y_true`` and ``y_prob`` arrays. Typically the output of ``eqb.slicer()``.
    :type data: Dict[str, Dict[str, np.ndarray]]

    :param curve_type: Type of curve to plot. Options are ``"roc"``, ``"pr"``, or ``"calibration"``.
    :type curve_type: str

    :param n_bins: Number of bins to use for calibration curves. Ignored for ROC and PR.
    :type n_bins: int

    :param decimal_places: Number of decimal places to show in curve labels (e.g., for AUC or Brier scores).
    :type decimal_places: int

    :param curve_kwgs: Optional dictionary mapping group names to curve styling parameters (e.g., ``color``, ``linestyle``).
    :type curve_kwgs: Dict[str, Dict[str, Union[str, float]]], optional

    :param line_kwgs: Optional styling for the reference line (e.g., diagonal in ROC or calibration).
    :type line_kwgs: Dict[str, Union[str, float]], optional

    :param title: Title of the entire figure.
    :type title: str

    :param filename: Filename prefix for saving the figure (without file extension).
    :type filename: str

    :param save_path: Directory path where the figure will be saved. If None, the plot is only displayed.
    :type save_path: str or None

    :param figsize: Tuple specifying the figure size in inches (width, height).
    :type figsize: Tuple[float, float]

    :param dpi: Resolution of the output figure in dots per inch.
    :type dpi: int

    :param subplots: Whether to generate a subplot per group. If False, all curves are overlaid.
    :type subplots: bool

    :param n_cols: Number of columns in the subplot grid.
    :type n_cols: int

    :param n_rows: Number of rows in the subplot grid. If None, it's inferred automatically.
    :type n_rows: int or None

    :param group: If set, plots only the specified group.
    :type group: str or None

    :param color_by_group: If True, assigns a different color to each group.
    :type color_by_group: bool

    :param exclude_groups: Optionally exclude specific groups by name or by minimum sample size.
    :type exclude_groups: Union[int, str, List[str], Set[str]]

    :param show_grid: Whether to display background gridlines in the plots.
    :type show_grid: bool

    :param lowess: Smoothing factor (0–1) for LOWESS calibration curves. Set to 0 to disable.
    :type lowess: float

    :param lowess_kwargs: Dictionary of additional styling arguments for LOWESS curves.
    :type lowess_kwargs: Dict[str, Union[str, float]], optional

    :param shade_area: Whether to fill the area beneath each curve (only for ROC and PR).
    :type shade_area: bool

    :param plot_hist: If True, displays a histogram of predicted probability counts beneath each calibration curve. Automatically enables ``subplots=True``.
    :type plot_hist: bool

    :returns: None. Displays or saves the plot depending on the ``save_path`` argument.
    :rtype: None

.. note::

    - When ``plot_hist=True``, each subplot includes a histogram showing how many predictions fall into each predicted probability bin. This is especially useful for interpreting calibration performance in regions with dense or sparse predictions.
    - LOWESS smoothing is useful for non-linear calibration curves or small sample groups.
    - When ``subplots=False`` and ``group=None``, all groups are overlaid on a single plot.
    - Setting both ``group`` and ``subplots=True`` will raise an error.


ROC AUC Curve
-----------------

The following code generates an ROC AUC curve comparing performance across racial groups. 
This visualization helps assess whether the model maintains similar true positive and 
false positive trade-offs across subpopulations.

By setting ``subplots=False``, the curves for each group are overlaid on a single plot, 
making disparities visually apparent. Groups with insufficient sample sizes or minimal 
representation can be excluded using the ``exclude_groups`` parameter, as shown below.

.. code:: python

    eqb.eq_plot_group_curves(
        sliced_race_data,
        curve_type="roc",
        title="ROC AUC by Race Group",
        figsize=(7, 7),
        decimal_places=2,
        subplots=False,
        exclude_groups=["Amer-Indian-Eskimo", "Other"]
    )



.. raw:: html

   <div class="no-click">

.. image:: ../assets/roc_auc_curves.png
   :alt: ROC AUC Curve
   :align: center
   :width: 600px

.. raw:: html

    <div style="height: 40px;"></div></div>

Precision-Recall Curves
-------------------------

.. code:: python

    eqb.eq_plot_group_curves(
        sliced_race_data,
        curve_type="pr",
        subplots=False,
        figsize=(7, 7),
        title="Precision-Recall by Race Group",
        exclude_groups=["Amer-Indian-Eskimo", "Other"]
    )

.. image:: ../assets/pr_curves.png
   :alt: Precision-Recall Curves
   :align: center
   :width: 600px

.. raw:: html

    <div style="height: 40px;"></div>


Calibration Plots
---------------------

Calibration plots compare predicted probabilities to actual outcomes, showing
how well the model's confidence aligns with observed frequencies. A perfectly
calibrated model will have a curve that closely follows the diagonal reference line.

The example below overlays calibration curves by racial group, using the same sliced data.
Groups with low representation are excluded to ensure stable and interpretable plots.

For additional context on the geometric intuition behind calibration curves, 
including how the area between the observed curve and the ideal diagonal can be 
interpreted, see the :ref:`Mathematical Framework <calibration_auc>` section. 
That section illustrates how integration under the curve provides a mathematical view of 
calibration performance.


Example 1 (Calibration Overlay)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    eqb.eq_plot_group_curves(
        sliced_race_data,
        curve_type="calibration",
        title="Calibration by Race Group",
        figsize=(7, 7),
        decimal_places=2,
        subplots=False,
        exclude_groups=["Amer-Indian-Eskimo", "Other"]
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/calibration_plot.png
   :alt: Calibration Plot Overlay
   :align: center
   :width: 600px

.. raw:: html

    <div style="height: 40px;"></div></div>

Example 2 (Calibration Subplots)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example builds on the previous one by showing individual calibration 
curves in separate subplots and enabling shaded areas beneath the curves. This 
layout improves visual clarity, especially when comparing many groups or when the 
overlaid version appears cluttered.

Setting ``shade_area=True`` fills the area under each calibration curve.
Subplots also help isolate each group’s performance, 
allowing easier inspection of group-specific trends.

.. code:: python

    eqb.eq_plot_group_curves(
        sliced_race_data,
        curve_type="calibration",
        title="Calibration by Race Group",
        figsize=(7, 7),
        decimal_places=2,
        subplots=True,
        shade_area=True,
        n_cols=3,
        exclude_groups=["Amer-Indian-Eskimo", "Other"]
    )



.. raw:: html

   <div class="no-click">

.. image:: ../assets/calibration_sub_plots.png
   :alt: Calibration Subplots
   :align: center
   :width: 600px

.. raw:: html

    <div style="height: 40px;"></div></div>



.. raw:: html

    <div style="height: 40px;"></div>

Example 3 (LOWESS Calibration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


This example demonstrates the use of **Locally Weighted Scatterplot Smoothing (LOWESS)** 
to fit a locally adaptive curve for calibration. This technique is helpful when 
calibration is non-linear or when jagged curves result from small group sizes or 
class imbalance.

.. note:: 

    Enable LOWESS smoothing by setting the ``lowess`` parameter to a float between 
    0 and 1, which controls the smoothing span. Additional styling can be applied 
    via ``lowess_kwargs``.

.. code:: python

    eqb.eq_plot_group_curves(
        sliced_race_data,
        curve_type="calibration",
        title="Calibration by Race Group (LOWESS Smoothing)",
        figsize=(7, 7),
        decimal_places=2,
        subplots=True,
        lowess=0.6,
        lowess_kwargs={"linestyle": "--", "linewidth": 2, "alpha": 0.6},
        n_cols=3,
        exclude_groups=["Amer-Indian-Eskimo", "Other"]
    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/lowess_calibration.png
   :alt: LOWESS-smoothed Calibration Subplots
   :align: center
   :width: 600px

.. raw:: html

    <div style="height: 40px;"></div></div>

LOWESS produces smoother and more flexible calibration curves compared to binning. 
It is particularly useful for identifying subtle trends in over or under-confidence 
across different segments of the population.

Example 4 (Calibration Histograms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


When ``plot_hist=True`` is enabled, the function displays a histogram of sample 
counts beneath the calibration curve for each group. Each bar shows how many 
predictions fall into a given probability bin (e.g., 0.0–0.1, 0.1–0.2). This is 
helpful for diagnosing whether calibration differences occur in well-populated 
regions of the probability spectrum or in sparse areas with few predictions.


.. note::

   Histograms are especially useful when interpreting overconfident or 
   underconfident predictions across different groups. Regions with sparse 
   histogram bars may also indicate model uncertainty or data scarcity in those 
   probability intervals.


.. code:: python

    eqb.eq_plot_group_curves(
        sliced_race_data,
        curve_type="calibration",
        title="Calibration by Race Group",
        n_bins=10,
        show_grid=False,
        exclude_groups=["Amer-Indian-Eskimo", "Other"]
        plot_hist=True,

    )

.. raw:: html

   <div class="no-click">

.. image:: ../assets/calibration_hist.png
   :alt: Calibration Curve with Histogram Overlay
   :align: center
   :width: 600px

.. raw:: html

    <div style="height: 40px;"></div></div>

The **histogram bars** at the base of the plot show how frequently predictions fall into each probability bin, grouped by demographic subgroup. This combined view helps validate whether deviations from the ideal diagonal are meaningful and well-supported by the underlying data.

For instance, if a group appears poorly calibrated in a region where very few predictions occur, the issue may be less impactful than one affecting densely populated bins. This visual diagnostic is especially valuable when auditing model behavior across real-world deployment scenarios.




.. [1] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.