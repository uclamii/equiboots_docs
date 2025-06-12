.. _equi_boots_class:

EquiBoots Class
===============

.. automodule:: equiboots.EquiBootsClass
   :members:
   :undoc-members:
   :exclude-members: bootstrap, grouper, groups_slicer, slicer, sample_group, analyze_statistical_significance
   :show-inheritance:


Overview
--------

The ``EquiBoots`` class provides tools for fairness-aware evaluation and bootstrapping of machine learning model predictions. It supports binary, multi-class, multi-label classification, and regression tasks, and enables group-based metric calculation, disparity analysis, and statistical significance testing.

Constructor
-----------

.. py:class:: EquiBoots(y_true, y_pred, fairness_df, fairness_vars, y_prob=None, seeds=[1,2,3,4,5,6,7,8,9,10], reference_groups=None, task="binary_classification", bootstrap_flag=False, num_bootstraps=10, boot_sample_size=100, balanced=True, stratify_by_outcome=False, group_min_size=10)
   :noindex:

   Initialize a new ``EquiBoots`` instance.

   **Parameters**

   - **y_true** (numpy.ndarray)  
     Ground truth labels.
   - **y_pred** (numpy.ndarray)  
     Predicted labels.
   - **fairness_df** (pandas.DataFrame)  
     DataFrame containing fairness variables.
   - **fairness_vars** (list of str)  
     Names of fairness variables.
   - **y_prob** (numpy.ndarray, optional)  
     Predicted probabilities.
   - **seeds** (list of int, optional)  
     Random seeds for bootstrapping.
   - **reference_groups** (list, optional)  
     Reference group for each fairness variable.
   - **task** (str)  
     One of ``binary_classification``, ``multi_class_classification``, ``multi_label_classification``, or ``regression``.
   - **bootstrap_flag** (bool)  
     Whether to perform bootstrapping.
   - **num_bootstraps** (int)  
     Number of bootstrap iterations.
   - **boot_sample_size** (int)  
     Size of each bootstrap sample.
   - **balanced** (bool)  
     If True, balance samples across groups; otherwise stratify by original proportions.
   - **stratify_by_outcome** (bool)  
     Stratify sampling by outcome label.
   - **group_min_size** (int)  
     Minimum group size (groups smaller than this are omitted).

   **Returns**

   None

Main Methods
------------

.. py:method:: grouper(groupings_vars)

   Groups data by the specified fairness variables and stores category indices.

   **Parameters**

   - **groupings_vars** (list of str)  
     Variables to group by.

   **Returns**

   None

.. py:method:: slicer(slicing_var)

   Slice ``y_true``, ``y_prob``, and ``y_pred`` by a single fairness variable.

   **Parameters**

   - **slicing_var** (str)  
     Variable name to slice by.

   **Returns**

   dict or list of dict  
     Sliced outputs.

.. py:method:: get_metrics(sliced_dict)

   Calculate performance metrics for each group.

   **Parameters**

   - **sliced_dict** (dict or list of dict)  
     Output of ``slicer``.

   **Returns**

   dict or list of dict  
     Metrics per group.

.. py:method:: calculate_disparities(metric_dict, var_name)

   Compute ratio disparities against the reference group.

   **Parameters**

   - **metric_dict** (dict or list of dict)  
     Group metrics.
   - **var_name** (str)  
     Fairness variable name.

   **Returns**

   dict or list of dict  
     Ratio disparities.

.. py:method:: calculate_differences(metric_dict, ref_var_name)

   Compute difference disparities against the reference group.

   **Parameters**

   - **metric_dict** (dict or list of dict)  
     Group metrics.
   - **ref_var_name** (str)  
     Reference group name.

   **Returns**

   dict or list of dict  
     Difference disparities.

.. py:method:: analyze_statistical_significance(metric_dict, var_name, test_config, differences=None)

   Perform significance testing on metric differences.

   **Parameters**

   - **metric_dict** (dict or list of dict)  
     Group metrics.
   - **var_name** (str)  
     Fairness variable name.
   - **test_config** (dict)  
     Statistical test configuration.
   - **differences** (dict, optional)  
     Precomputed differences.

   **Returns**

   dict  
     Statistical test results per group.

.. py:method:: set_fix_seeds(seeds)

   Set fixed random seeds for reproducibility.

   **Parameters**

   - **seeds** (list of int)  
     Seeds to apply.

   **Returns**

   None

.. py:method:: list_available_tests()

   List the available statistical tests.

   **Returns**

   dict  
     Test names and descriptions.

.. py:method:: list_adjustment_methods()

   List the available p-value adjustment methods.

   **Returns**

   dict  
     Adjustment methods.

Non-Main/Internal Methods
-------------------------

.. py:method:: set_reference_groups(reference_groups)

   Set or infer reference groups for fairness variables.

   **Parameters**

   - **reference_groups** (list)  
     Reference groups to use.

   **Returns**

   None

.. py:method:: check_task(task)

   Validate the task type.

   **Parameters**

   - **task** (str)  
     Task name.

   **Returns**

   None

.. py:method:: check_classification_task(task)

   Ensure the task is a classification type.

   **Parameters**

   - **task** (str)  
     Task name.

   **Returns**

   None

.. py:method:: check_fairness_vars(fairness_vars)

   Validate the fairness variables input.

   **Parameters**

   - **fairness_vars** (list of str)  
     Variables to validate.

   **Returns**

   None

.. py:method:: check_group_size(group, cat, var)

   Verify minimum size for a group.

   **Parameters**

   - **group**  
     Group data.
   - **cat**  
     Category name.
   - **var**  
     Variable name.

   **Returns**

   None

.. py:method:: check_group_empty(sampled_group, cat, var)

   Check if a sampled group is empty.

   **Parameters**

   - **sampled_group**  
     Group data.
   - **cat**  
     Category name.
   - **var**  
     Variable name.

   **Returns**

   None

.. py:method:: sample_group(group, n_categories, indx, sample_size, seeds, balanced)

   Draw bootstrap or stratified samples.

   **Parameters**

   - **group**  
     Group data.
   - **n_categories** (int)  
     Number of categories.
   - **indx**  
     Indices of data.
   - **sample_size** (int)  
     Bootstrap sample size.
   - **seeds** (list of int)  
     Random seeds.
   - **balanced** (bool)  
     Balance flag.

   **Returns**

   The sampled group data.

.. py:method:: groups_slicer(groups, slicing_var)

   Slice data into categories for a given variable.

   **Parameters**

   - **groups**  
     Group index mapping.
   - **slicing_var** (str)  
     Variable name.

   **Returns**

   dict or list of dict  
     Sliced data.

.. py:method:: get_groups_metrics(sliced_dict)

   Calculate metrics for each group.

   **Parameters**

   - **sliced_dict** (dict or list of dict)  
     Sliced data.

   **Returns**

   dict or list of dict  
     Metrics per group.

.. py:method:: calculate_groups_disparities(metric_dict, var_name)

   Compute ratio disparities for each group.

   **Parameters**

   - **metric_dict** (dict or list of dict)  
     Group metrics.
   - **var_name** (str)  
     Fairness variable name.

   **Returns**

   dict or list of dict  
     Ratio disparities.

.. py:method:: calculate_groups_differences(metric_dict, ref_var_name)

   Compute difference disparities for each group.

   **Parameters**

   - **metric_dict** (dict or list of dict)  
     Group metrics.
   - **ref_var_name** (str)  
     Reference group name.

   **Returns**

   dict or list of dict  
     Difference disparities.

Example Usage
--------------------

Below are two dummy examples demonstrating how to use the ``EquiBoots`` class: one **without** bootstrapping and one **with** bootstrapping.

For more detailed examples, refer to that Colab notebook or `py_scripts/testingscript.py`.

Point Estimates Without Bootstrapping
-------------------------------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from equiboots import EquiBoots

   # Example data
   y_true = np.array([0, 1, 1, 0, 1])
   y_prob = np.array([0.2, 0.8, 0.7, 0.4, 0.9])
   y_pred = np.array([0, 1, 1, 0, 1])
   fairness_df = pd.DataFrame({
       "race": ["A", "B", "A", "B", "A"],
       "sex": ["M", "F", "F", "M", "F"]
   })

   eq = EquiBoots(
       y_true=y_true,
       y_prob=y_prob,
       y_pred=y_pred,
       fairness_df=fairness_df,
       fairness_vars=["race", "sex"],
       task="binary_classification",
       bootstrap_flag=False
   )

   eq.grouper(groupings_vars=["race"])
   sliced = eq.slicer("race")
   metrics = eq.get_metrics(sliced)
   disparities = eq.calculate_disparities(metrics, "race")

   print("Metrics by group:", metrics)
   print("Disparities:", disparities)

With Bootstrapping
------------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   from equiboots import EquiBoots

   # Example data
   y_true = np.array([0, 1, 1, 0, 1])
   y_prob = np.array([0.2, 0.8, 0.7, 0.4, 0.9])
   y_pred = np.array([0, 1, 1, 0, 1])
   fairness_df = pd.DataFrame({
       "race": ["A", "B", "A", "B", "A"],
       "sex": ["M", "F", "F", "M", "F"]
   })

   eq = EquiBoots(
       y_true=y_true,
       y_prob=y_prob,
       y_pred=y_pred,
       fairness_df=fairness_df,
       fairness_vars=["race", "sex"],
       task="binary_classification",
       bootstrap_flag=True,
       num_bootstraps=5,
       boot_sample_size=5
   )

   eq.grouper(groupings_vars=["race"])
   sliced = eq.slicer("race")
   metrics = eq.get_metrics(sliced)
   disparities = eq.calculate_disparities(metrics, "race")

   print("Metrics by group (bootstrapped):", metrics)
   print("Disparities (bootstrapped):", disparities)

StatisticalTester
=================

Module: ``equiboots.StatisticalTester``

Overview
--------

This module provides statistical significance testing utilities, including bootstrapped and chi-square tests, with support for multiple comparison corrections and effect size calculations.

Classes
-------

StatTestResult
^^^^^^^^^^^^^^

.. autoclass:: equiboots.StatisticalTester.StatTestResult
    :members:
    :show-inheritance:

StatisticalTester
^^^^^^^^^^^^^^^^^

.. autoclass:: equiboots.StatisticalTester.StatisticalTester
    :members:
    :show-inheritance:

Function Signatures
-------------------

.. py:class:: StatTestResult(statistic: float, p_value: float, is_significant: bool, test_name: str, critical_value: Optional[float] = None, effect_size: Optional[float] = None, confidence_interval: Optional[Tuple[float, float]] = None)

    Stores statistical test results including test statistic, p-value, and significance.

.. py:class:: StatisticalTester()

    Performs statistical significance testing on metrics with support for various tests and data types.

    .. py:method:: _bootstrap_test(data: List[float], config: dict) -> StatTestResult

    .. py:method:: get_ci_bounds(config: dict) -> tuple

    .. py:method:: calc_p_value_bootstrap(data: list, config: dict) -> float

    .. py:method:: _chi_square_test(metrics: Dict[str, Any], config: Dict[str, Any]) -> StatTestResult

    .. py:method:: _calculate_effect_size(metrics: Dict) -> float

    .. py:method:: _adjust_p_values(results: Dict[str, Dict[str, StatTestResult]], method: str, alpha: float, boot: bool = False) -> Dict[str, Dict[str, StatTestResult]]

    .. py:method:: analyze_metrics(metrics_data: Union[Dict, List[Dict]], reference_group: str, test_config: Dict[str, Any], task: Optional[str] = None, differences: Optional[dict] = None) -> Dict[str, Dict[str, StatTestResult]]

    .. py:method:: adjusting_p_vals(config, results)

    .. py:method:: _validate_config(config: Dict[str, Any])

    .. py:method:: cohens_d(data_1, data_2)

    .. py:method:: _analyze_single_metrics(metrics: Dict, reference_group: str, config: Dict[str, Any]) -> Dict[str, Dict[str, StatTestResult]]

    .. py:method:: _analyze_bootstrapped_metrics(metrics_diff: list[Dict], reference_group: str, config: Dict[str, Any]) -> Dict[str, Dict[str, StatTestResult]]

Usage Example
-------------

.. code-block:: python

    from equiboots.StatisticalTester import StatisticalTester

    tester = StatisticalTester()
    config = {
        "test_type": "chi_square",
        "alpha": 0.05,
        "adjust_method": "bonferroni"
    }
    metrics = {
        "group1": {"TP": 10, "FP": 5, "TN": 20, "FN": 2},
        "group2": {"TP": 8, "FP": 7, "TN": 18, "FN": 4}
    }
    results = tester.analyze_metrics(metrics, reference_group="group1", test_config=config, task="binary_classification")
    for group, result in results.items():
        print(f"{group}: p-value={result.p_value}, significant={result.is_significant}")
