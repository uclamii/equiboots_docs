
.. _getting_started:   

.. raw:: html

   <div class="no-click">

.. image:: ../assets/EquiBoots.png
   :alt: EquiBoots Logo
   :align: left
   :width: 300px

.. raw:: html
   
   <div style="height: 130px;"></div></div>


Welcome to the EquiBoots Documentation!
==========================================

.. note::
   This documentation is for ``equiboots`` version ``0.0.0a13``.

EquiBoots is a fairness-aware model evaluation toolkit for auditing performance 
disparities across demographic groups in machine learning models. It provides robust, 
bootstrapped evaluation metrics for binary, multi-class, and multi-label classification 
tasks, as well as regression models.

The library supports:

- Group-wise performance slicing
- Fairness diagnostics and disparity metrics
- Confidence intervals via bootstrapping
- Customizable and publication-ready visualizations
- Statistical tests to assess performance differences

EquiBoots is suited for applications in clinical, social, and 
policy contexts: domains where transparency, bias mitigation, and equitable outcomes 
are essential for responsible AI/ML deployment.

Project Links
---------------

1. `PyPI Page <https://pypi.org/project/equiboots/>`_  

2. `GitHub Repository <https://github.com/uclamii/equiboots>`_

3. `Classification: Google Colab Example Notebook <https://colab.research.google.com/drive/1BrPCOO84nRYssX9JvQcAAC9azVmmGKjB>`_

4. `Regression: Google Colab Example Notebook <https://colab.research.google.com/drive/1VTZLCeuSwTtH9gbHPkCeXm-ovTqFmv6F#scrollTo=-iwzgLeE2_Aj>`_

Prerequisites
-------------
Before you install ``equiboots``, ensure your system meets the following requirements:

- **Python** (version ``3.7.4`` or higher)

Additionally, ``equiboots`` depends on the following packages, which will be automatically installed when you install ``equiboots``:

- ``matplotlib``: version ``3.5.3`` or higher, but capped at ``3.10.1``
- ``numpy``: version ``1.21.6`` or higher, but capped at ``2.2.4``
- ``pandas``: version ``1.3.5`` or higher, but capped at ``2.2.3``
- ``scikit-learn``: version ``1.0.2`` or higher, but capped at ``1.5.2``
- ``scipy``: version ``1.8.0`` or higher, but capped at ``1.15.2``
- ``seaborn``: version ``0.11.2`` or higher, but capped at ``0.13.2``
- ``statsmodels``: version ``0.13`` or higher, but capped at ``0.14.4``
- ``tqdm```: version ``4.66.4`` or higher, but capped below ``4.67.1``

.. _installation:

Installation
-------------

You can install ``equiboots`` directly from PyPI:

.. code-block:: python

    pip install equiboots

Description
--------------

This guide provides detailed instructions and examples for using the functions 
provided in the ``equiboots`` library and how to use them effectively in your projects.


----

Table of Contents
===================

.. toctree::
   :maxdepth: 4
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 4
   :caption: Point Estimate Metrics

   point_estimate_metrics

.. toctree::
   :maxdepth: 4
   :caption: Bootstrap Estimate Metrics

   bootstrapped_estimates


.. toctree::
   :maxdepth: 4
   :caption: Classes, Attributes, & Methods

   equiboots_stats_classes

.. toctree::
   :maxdepth: 4
   :caption: From Model to Prediction

   model_to_pred

.. toctree::
   :maxdepth: 4
   :caption: Mathematical Framework

   mathematical_framework

.. toctree::
   :maxdepth: 4
   :caption: iPython Notebooks

   ipynb_notebooks

.. toctree::
   :maxdepth: 4
   :caption: About EquiBoots

   acknowledgements
   contributors
   citations
   changelog
   references


