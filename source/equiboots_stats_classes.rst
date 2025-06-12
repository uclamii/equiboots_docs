.. _equiboots_class:



EquiBoots Class
========================

.. autoclass:: EquiBoots
   :members:
   :undoc-members:
   :show-inheritance:

Overview
~~~~~~~~~~~~~

The ``EquiBoots`` class offers a structured framework for fairness auditing in machine learning. 
It supports the evaluation of performance metrics across subgroups defined by sensitive attributes 
such as race, sex, or age, with optional bootstrapping to assess metric stability and disparity.


Constructor
~~~~~~~~~~~~~~~~~~

.. py:class:: EquiBoots(y_true, y_pred, fairness_df, fairness_vars, y_prob=None, seeds=[1,2,3,4,5,6,7,8,9,10], reference_groups=None, task='binary_classification', bootstrap_flag=False, num_bootstraps=10, boot_sample_size=100, balanced=True)

   :param y_true: Array of ground truth labels.
   :type y_true: np.array

   :param y_pred: Array of predicted labels (discrete).
   :type y_pred: np.array

   :param fairness_df: DataFrame containing subgroup identifiers (e.g., race, sex).
   :type fairness_df: pandas.DataFrame

   :param fairness_vars: List of column names in ``fairness_df`` to audit fairness across.
   :type fairness_vars: list

   :param y_prob: Optional array of predicted probabilities or scores.
   :type y_prob: np.array or None

   :param seeds: Random seeds used to generate bootstraps.
   :type seeds: list

   :param reference_groups: Optional list of reference groups, one per fairness variable.
   :type reference_groups: list or None

   :param task: Task type, one of ``'binary_classification'``, ``'multi_class'``, ``'multi_label'``, or ``'regression'``.
   :type task: str

   :param bootstrap_flag: Whether to perform bootstrapped evaluation.
   :type bootstrap_flag: bool

   :param num_bootstraps: Number of bootstrap replicates.
   :type num_bootstraps: int

   :param boot_sample_size: Size of each bootstrap sample.
   :type boot_sample_size: int

   :param balanced: Whether to enforce balanced sampling across subgroups during bootstrapping.
   :type balanced: bool

Methods
~~~~~~~~~~~~~

.. py:method:: get_grouped_data()

   Slice input data by each fairness variable and group, returning a dictionary structure containing ``y_true``, ``y_pred``, and optionally ``y_prob`` for each subgroup.

   :returns: Dict of nested group-wise arrays.
   :rtype: dict

.. py:method:: run_bootstrap()

   Perform bootstrapped sampling and compute metrics across all specified groups. Required if ``bootstrap_flag`` is True.

   :returns: Bootstrapped metrics per group per seed.
   :rtype: dict

.. py:method:: summarize_metrics()

   Compute performance metrics by group (with or without bootstrapping).

   :returns: Dictionary or DataFrame of grouped performance metrics.
   :rtype: dict or pandas.DataFrame

.. py:method:: compute_disparities()

   Compare each subgroup against a reference group for each fairness variable. Computes metric-level disparities.

   :returns: Dictionary of disparities across metrics and groups.
   :rtype: dict

.. py:method:: summarize_boot_disparities()

   Compute disparities across bootstrap replicates and return confidence intervals.

   :returns: Nested dictionary of disparity distributions and confidence intervals.
   :rtype: dict

.. py:method:: get_confidence_intervals(metric_dict)

   Given a dictionary of bootstrapped metric results, compute 95% confidence intervals.

   :param metric_dict: Bootstrapped results keyed by group and metric.
   :type metric_dict: dict

   :returns: Confidence intervals per metric per group.
   :rtype: dict

.. py:method:: to_dataframe(metric_dict)

   Utility method to convert metric dictionaries into tabular format.

   :param metric_dict: Dictionary of results keyed by group and metric.
   :type metric_dict: dict

   :returns: DataFrame of results.
   :rtype: pandas.DataFrame

StatisticalTester Class
===============================

Overview
~~~~~~~~~~~~~~

The ``StatisticalTester`` class provides a flexible framework for statistical significance testing 
across subgroup metrics in machine learning fairness audits. It supports both parametric and 
non-parametric tests, p-value adjustments, and bootstrapped confidence interval estimation.

Classes
~~~~~~~~~~~~~~~~~~~~

.. py:class:: StatTestResult

   Stores the result of a statistical test.

   :param statistic: Test statistic value.
   :type statistic: float

   :param p_value: Computed p-value for the test.
   :type p_value: float

   :param is_significant: Boolean indicating whether the test result is statistically significant.
   :type is_significant: bool

   :param test_name: Name of the statistical test.
   :type test_name: str

   :param critical_value: Optional critical value used for decision-making.
   :type critical_value: float, optional

   :param effect_size: Optional effect size (e.g., Cramér's V or Cohen's d).
   :type effect_size: float, optional

   :param confidence_interval: Optional confidence interval for the test statistic.
   :type confidence_interval: tuple(float, float), optional

.. py:class:: StatisticalTester

   Performs statistical testing with support for bootstrapping, chi-square, p-value correction, 
   and effect size estimation.

   .. py:attribute:: AVAILABLE_TESTS
      :type: dict
      :value: {"chi_square": "Chi-square test", "bootstrap_test": "Bootstrap test"}

   .. py:attribute:: ADJUSTMENT_METHODS
      :type: dict
      :value: {"bonferroni", "fdr_bh", "holm", "none"}

   .. py:method:: __init__()

      Initializes test method mappings.

   .. py:method:: analyze_metrics(metrics_data, reference_group, test_config, task=None, differences=None)
      
      Main interface to analyze group metrics for significance relative to a reference group.

      :param metrics_data: Metric data in either dict or list of dicts format.
      :param reference_group: The baseline group to compare others against.
      :param test_config: Dictionary of test parameters (e.g., alpha, test_type).
      :param task: Optional task type (e.g., "binary_classification").
      :param differences: Optional differences dictionary for bootstrapped metrics.
      :returns: Dictionary of test results.
      :rtype: dict

   .. py:method:: get_ci_bounds(config)

      Returns lower and upper percentile bounds for CI based on test type.

   .. py:method:: _bootstrap_test(data, config)

      Performs a bootstrap significance test with CI and effect direction.

   .. py:method:: _chi_square_test(metrics, config)

      Performs a chi-square test on a contingency table of confusion matrix values.

   .. py:method:: _calculate_effect_size(metrics)

      Computes Cramér's V from a contingency table.

   .. py:method:: calc_p_value_bootstrap(data, config)

      Calculates a bootstrap-based p-value assuming symmetry around 0.

   .. py:method:: _adjust_p_values(results, method, alpha, boot=False)

      Adjusts p-values for multiple comparisons.

   .. py:method:: adjusting_p_vals(config, results)

      Wrapper for selecting and running the appropriate p-value adjustment method.

   .. py:method:: cohens_d(data_1, data_2)

      Computes Cohen's d between two arrays.

   .. py:method:: _analyze_single_metrics(metrics, reference_group, config)

      Performs significance testing on confusion matrix values without bootstrapping.

   .. py:method:: _analyze_bootstrapped_metrics(metrics_diff, reference_group, config)

      Runs bootstrapped testing on performance metric differences by group.

   .. py:method:: _validate_config(config)

      Validates required config keys and test types.
