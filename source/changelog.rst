.. raw:: html

   <div class="no-click">

.. image:: ../assets/EquiBoots.png
   :alt: EquiBoots Logo
   :align: left
   :width: 300px

.. raw:: html
   
   <div style="height: 130px;"></div></div>

Changelog
===============

`Version 0.0.1a1`_
----------------------

- minor change to address author consistency and updated documentation link within ``__init__.py``

`Version 0.0.1a`_
----------------------

**What's Changed**

* Add ``plot_effect_sizes`` helper, update adult income notebook, tests, ``.gitignore``, and requirements by @lshpaner in https://github.com/uclamii/equiboots/pull/61
* (+) forest plot; (-) duplicated lines by @lshpaner in https://github.com/uclamii/equiboots/pull/62
* Adding point estimate forest plots by @elemets in https://github.com/uclamii/equiboots/pull/63


**Full Changelog**: https://github.com/uclamii/equiboots/compare/0.0.0a10...0.0.1a

`Version 0.0.0a10`_
----------------------

Add histogram-only mode to calibration curves (``plot_hist`` flag)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we introduce a new boolean parameter, ``plot_hist``, to ``eq_plot_group_curves`` (and propagate it down into ``_plot_group_curve_ax``). When ``plot_hist`=True``, we:

- Automatically switch into ``"subplots"`` mode (one axis per group), regardless of the caller’s subplots setting.
- Skip the regular calibration‐curve drawing and render a simple histogram of ``y_prob`` instead.
- Color each histogram with the exact same per‐group color from ``curve_kwgs`` (or the default color map).

.. code:: python 

   eqb.eq_plot_group_curves(
      sliced_data,
      curve_type="calibration",
      title="Calibration by Race Group",
      n_bins=10,
      show_grid=False,
      plot_hist=True,
      # subplots=True,
      # exclude_groups="white",
   )

Handle Seaborn ``< 0.13.2`` legend errors for boxplot/violinplot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This patch adds a runtime version check so that, on Seaborn 0.12.x, we don’t pass the unsupported legend kwarg and then explicitly remove the auto‐drawn legend. On 0.13.2+ we keep using ``legend=False`` as before.

- We import ``version`` from ``packaging`` and check ``sns.__version__`` once at module load.
- Inside the loop we only add ``legend=False`` when Seaborn ≥ 0.13.2.
- For older versions we catch the ``TypeError``, retry without ``legend``, then drop any stray legend.

**Unit Test Updates:** 

- Switch tests from ``plot_kind`` to ``plot_type`` based on correct code.
- Add seaborn version‐specific legend tests 

Add ``lowess_kwargs`` support & show LOWESS AUC in legend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Introduce new ``lowess_kwargs`` arg (merged with curve_kwargs --> defaults)
- Compute lowess_auc = calibration_auc(x_s, y_s) and plot LOWESS with:

.. code:: python

   ax.plot(
      x_s, y_s,
      label=f"LOWESS (AUC={lowess_auc:.3f})",
      **smooth_kwargs
   )

- No breaking changes; existing calls without ``lowess_kwargs`` continue to work.



`Version 0.0.0a9`_
----------------------

What's Changed
------------------

* Control Threshold Line, Multiple ``y_lims`` inside ``eq_group_metrics_point_plot`` by @lshpaner in https://github.com/uclamii/equiboots/pull/52

**Full Changelog**: https://github.com/uclamii/equiboots/compare/0.0.0a8...0.0.0a9


`Version 0.0.0a8`_
----------------------

- ``tqdm`` added to bootstrap
- reduced the number of decimal places on the tables to 3 dp
- added p-value calc for when bootstrap number under 5000

`Version 0.0.0a7`_
----------------------

* Area calib ls by @lshpaner in https://github.com/uclamii/equiboots/pull/50
* Statistically Significant Tables and Plots by @elemets in https://github.com/uclamii/equiboots/pull/48


**Full Changelog**: https://github.com/uclamii/equiboots/compare/0.0.0a6...0.0.0a7


`Version 0.0.0a6`_
----------------------

**Full Changelog**: https://github.com/uclamii/equiboots/compare/0.0.0a5...0.0.0a6

- Fixed bug where y_prob was a required variable. Should now support regression properly.

`Version 0.0.0a5`_
----------------------

* Added ``statsmodels`` dependency; updated ``README`` and ``pyproject.toml`` accordingly;

Otherwise, see the following substantive changes from 0.0.0a4:

* Refactor and Expand Grouped Model Evaluation Plots with Residuals, Bootstrapped Curves, and Disparity Metrics by @lshpaner in https://github.com/uclamii/equiboots/pull/22
* Switched raise to warn + updated tests and requirements by @lshpaner in https://github.com/uclamii/equiboots/pull/24
* Pp filter low sample by @panas89 in https://github.com/uclamii/equiboots/pull/25
* Enhance Plotting Functions with Dynamic Y-Axis Limits, Pass/Fail Logic, New Point Plot Function, and Tests by @lshpaner in https://github.com/uclamii/equiboots/pull/26
* Renamed functions; updated nb; updates tests/unittests by @lshpaner in https://github.com/uclamii/equiboots/pull/29
* Metrics DataFrame Function by @lshpaner in https://github.com/uclamii/equiboots/pull/32
* Notebook refine sklearn except ls by @lshpaner in https://github.com/uclamii/equiboots/pull/33
* add support for statistical tests by @arahrooh31 in https://github.com/uclamii/equiboots/pull/23
* added differences of metrics by @panas89 in https://github.com/uclamii/equiboots/pull/39
* Update ``_adjust_p_values`` to Support Dynamic P-Value Adjustments by @lshpaner in https://github.com/uclamii/equiboots/pull/42
* Effect Size Fix by @elemets in https://github.com/uclamii/equiboots/pull/41
* Pp tests by @panas89 in https://github.com/uclamii/equiboots/pull/43
* Bootstrap tests by @elemets in https://github.com/uclamii/equiboots/pull/47
* Refactor Plotting Module by @lshpaner in https://github.com/uclamii/equiboots/pull/40

New Contributor
~~~~~~~~~~~~~~~~~~~~~~

* @arahrooh31 made their first contribution in https://github.com/uclamii/equiboots/pull/23

**Full Changelog**: https://github.com/uclamii/equiboots/compare/0.0.0a3...0.0.0a4

**Full Changelog**: https://github.com/uclamii/equiboots/compare/0.0.0a4...0.0.0a5


`Version 0.0.0a4`_
----------------------

* Refactor and Expand Grouped Model Evaluation Plots with Residuals, Bootstrapped Curves, and Disparity Metrics by @lshpaner in https://github.com/uclamii/equiboots/pull/22
* Switched raise to warn + updated tests and requirements by @lshpaner in https://github.com/uclamii/equiboots/pull/24
* Pp filter low sample by @panas89 in https://github.com/uclamii/equiboots/pull/25
* Enhance Plotting Functions with Dynamic Y-Axis Limits, Pass/Fail Logic, New Point Plot Function, and Tests by @lshpaner in https://github.com/uclamii/equiboots/pull/26
* Renamed functions; updated nb; updates tests/unittests by @lshpaner in https://github.com/uclamii/equiboots/pull/29
* Metrics DataFrame Function by @lshpaner in https://github.com/uclamii/equiboots/pull/32
* Notebook refine sklearn except ls by @lshpaner in https://github.com/uclamii/equiboots/pull/33
* add support for statistical tests by @arahrooh31 in https://github.com/uclamii/equiboots/pull/23
* added differences of metrics by @panas89 in https://github.com/uclamii/equiboots/pull/39
* Update ``_adjust_p_values`` to Support Dynamic P-Value Adjustments by @lshpaner in https://github.com/uclamii/equiboots/pull/42
* Effect Size Fix by @elemets in https://github.com/uclamii/equiboots/pull/41
* Pp tests by @panas89 in https://github.com/uclamii/equiboots/pull/43
* Bootstrap tests by @elemets in https://github.com/uclamii/equiboots/pull/47
* Refactor Plotting Module by @lshpaner in https://github.com/uclamii/equiboots/pull/40

New Contributor
~~~~~~~~~~~~~~~~~~~~~~
* @arahrooh31 made their first contribution in https://github.com/uclamii/equiboots/pull/23

**Full Changelog**: https://github.com/uclamii/equiboots/compare/0.0.0a3...0.0.0a4


`Version 0.0.0a3`_
----------------------

* added stratify by outcome for classification by @panas89 in https://github.com/uclamii/equiboots/pull/14
* Add Centroid Overlay, Group Stats, and Legend Enhancements to ``eq_plot_residuals_by_group``, (+) Function Enhancements by @lshpaner in https://github.com/uclamii/equiboots/pull/17


**Full Changelog**: https://github.com/uclamii/equiboots/compare/0.0.0a2...0.0.0a3

`Version 0.0.0a2`_
----------------------

- Added Zenodo DOI badge for citation and reproducibility.
- Cleaned up ``README.md``:
  - Aligned and formatted dependencies for improved readability.
  - Updated outdated or broken links.
  - Included Zenodo citation section with properly formatted reference.


`Version 0.0.0a1`_
----------------------
* Pp metrics by @panas89 in https://github.com/uclamii/equiboots/pull/1
* Updated Requirements by @lshpaner in https://github.com/uclamii/equiboots/pull/2
* Initialising Project (Grouper, Slicer) by @elemets in https://github.com/uclamii/equiboots/pull/3
* added calibration plot by @panas89 in https://github.com/uclamii/equiboots/pull/5
* Add Precision-Recall Curve and Enhance Calibration Curve Plot by @lshpaner in https://github.com/uclamii/equiboots/pull/6
* Disparities and reference groups by @elemets in https://github.com/uclamii/equiboots/pull/7
* Adding metrics and test for metrics by @elemets in https://github.com/uclamii/equiboots/pull/4
* Add Support for Reproducible Seeds in ``EquiBoots`` by @lshpaner in https://github.com/uclamii/equiboots/pull/8
* Pp bootstrap by @panas89 in https://github.com/uclamii/equiboots/pull/9
* (+) ``eq_disparity_metrics_plot``, linted code, (-) unused imports by @lshpaner in https://github.com/uclamii/equiboots/pull/10
* Add Unit Tests, Package Refactor, and Import Fixes by @lshpaner in https://github.com/uclamii/equiboots/pull/11
* added multitask support and validated reprodu. results by @panas89 in https://github.com/uclamii/equiboots/pull/12
* Add Bootstrapped Grouped Visualization for ROC, PR, and Calibration Curves with Confidence Intervals by @lshpaner in https://github.com/uclamii/equiboots/pull/13

New Contributors
~~~~~~~~~~~~~~~~~~~~~~
* @panas89 made their first contribution in https://github.com/uclamii/equiboots/pull/1
* @lshpaner made their first contribution in https://github.com/uclamii/equiboots/pull/2
* @elemets made their first contribution in https://github.com/uclamii/equiboots/pull/3

**Full Changelog**: https://github.com/uclamii/equiboots/commits/0.0.0a1