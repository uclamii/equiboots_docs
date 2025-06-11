.. _model_to_pred:   

Preparing Model Outputs for Fairness Analysis
=====================================================================


Before conducting any fairness or bias audits with EquiBoots, we must first train 
a model and extract the necessary components: predicted labels, predicted probabilities, 
true labels, and sensitive attributes. This section walks through the process using 
the Adult Income dataset [1]_, a popular dataset from the UCI Machine Learning Repository [2]_.


Step 1: Install and Import Dependencies
-----------------------------------------

We begin by installing the ucimlrepo package and importing the necessary Python 
libraries for data handling, preprocessing, modeling, and evaluation.

.. code:: bash

    pip install ucimlrepo

.. code:: python

    ## Import Necessary Libraries

    from ucimlrepo import fetch_ucirepo
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report

Step 2: Load the Dataset
-----------------------------

We fetch the Adult Income dataset directly from the UCI repository using ucimlrepo. 
This returns a dataset with features and a target variable indicating whether 
an individual's income is above or below $50K.

.. code:: python

    ## Fetch Dataset
    adult = fetch_ucirepo(id=2)
    adult = adult.data.features.join(adult.data.targets, how="inner")

Step 3: Clean the Data
---------------------------

Missing values are removed to simplify modeling. We also create a backup 
copy of the cleaned dataset.

**a. Drop missing values**

.. code:: python

    adult.dropna(inplace=True)

**b. Copy DataFrame for posterity**

.. code:: python

    df = adult.copy()


Step 4: Encode the Target Variable
-------------------------------------

The target column ``'income'`` is a string. We convert it into a binary format: 0 for ``<=50K`` and 1 for ``>50K``.
    
.. code:: python

    def outcome_merge(val):

        if val == '<=50K' or val == '<=50K.':
            return 0
        else:
            return 1

.. code:: python

    df['income'] = df['income'].apply(outcome_merge)

Step 5: Prepare Features and Labels
--------------------------------------

We split the dataset into features ``X`` and labels ``y``. Categorical variables 
are encoded as pandas category types to be handled natively by XGBoost.

.. code:: python

    X = df.drop("income", axis=1)
    y = df["income"]

    for col in X.columns:
    if isinstance(X[col], object):
        X[col] = X[col].astype("category")

Step 6: Train-Test Split
----------------------------

We split the data into training and test sets using an 80/20 ratio.

.. code:: python

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )


Step 7: Train the XGBoost Model
------------------------------------

We fit an XGBoost classifier on the training data. 

.. code:: python

    model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        enable_categorical=True
    )
    model.fit(X_train, y_train)

.. note::    

    Note that ``enable_categorical=True`` is used to support categorical columns directly.

Step 8: Generate Predictions and Evaluate
-------------------------------------------

We obtain both predicted class labels (``y_pred``) and predicted probabilities 
(``y_prob``) from the model and print a standard classification report to evaluate 
performance.

.. code:: python

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    print(classification_report(y_test, y_pred))

.. code:: text

                precision    recall  f1-score   support

            0        0.88      0.94      0.91      7170
            1        0.76      0.63      0.69      2355

     accuracy                            0.86      9525
    macro avg        0.82      0.78      0.80      9525
 weighted avg        0.85      0.86      0.86      9525


Step 9: Extract Model Outputs
----------------------------------

We now extract the model’s predicted class labels, predicted probabilities for 
the positive class (income > 50K), and convert the true labels to a NumPy array. 
These are the core inputs required by EquiBoots.

.. code:: python


    ## Get predicted class labels (0 or 1)
    y_pred = model.predict(X_test)

    ## Get predicted probabilities for class 1 (income > 50K)
    y_prob = model.predict_proba(X_test)[:, 1]

    ## Convert ground truth labels to NumPy array
    y_true = y_test.to_numpy()


.. [1] Kohavi, R. (1996). *Census Income*. UCI Machine Learning Repository. `https://doi.org/10.24432/C5GP7S <https://doi.org/10.24432/C5GP7S>`_.
.. [2] Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. `https://archive.ics.uci.edu <https://archive.ics.uci.edu>`_.