ibreakdown
==========
.. image:: https://travis-ci.com/jettify/ibreakdown.svg?branch=master
    :target: https://travis-ci.com/jettify/ibreakdown
.. image:: https://codecov.io/gh/jettify/ibreakdown/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jettify/ibreakdown
.. image:: https://img.shields.io/pypi/pyversions/ibreakdown.svg
    :target: https://pypi.org/project/ibreakdown
.. image:: https://img.shields.io/pypi/v/ibreakdown.svg
    :target: https://pypi.python.org/pypi/ibreakdown


**ibreakdown** is model agnostic predictions explainer with interactions support,
library can show contribution of each feature in your prediction value.

**SHAP** or **LIME** consider only local additive feature attributions, when
**ibreakdown** also evaluates local feature interactions.

Algorithm
=========

Algorithm is based on ideas describe in paper *"iBreakDown: Uncertainty of Model
Explanations for Non-additive Predictive Models"* https://arxiv.org/abs/1903.11420 and
reference implementation in **R** (iBreakDown_)

Intuition behind algorithm is following:

  ::

   The algorithm works in a similar spirit as SHAP or Break Down but is not
   restricted to additive effects. The intuition is the following:

   1. Calculate a single-step additive contribution for each feature.
   2. Calculate a single-step contribution for every pair of features. Subtract additive contribution to assess the interaction specific contribution.
   3. Order interaction effects and additive effects in a list that is used to determine sequential contributions.

   This simple intuition may be generalized into higher order interactions.

In depth explanation can be found in algorithm authors free book:
*Predictive Models: Explore, Explain, and Debug* https://pbiecek.github.io/PM_VEE/iBreakDown.html


Simple example
--------------

.. code:: python

    # model = RandomForestClassifier(...)
    explainer = ClassificationExplainer(model)
    classes = ['Deceased', 'Survived']
    explainer.fit(X_train, columns, classes)
    exp = explainer.explain(observation)
    exp.print()

Please check full Titanic example here: https://github.com/jettify/ibreakdown/blob/master/examples/titanic.py

.. code::

   +------------------------------------+-----------------+--------------------+--------------------+
   | Feature Name                       | Feature Value   |   Contrib:Deceased |   Contrib:Survived |
   +------------------------------------+-----------------+--------------------+--------------------|
   | intercept                          |                 |          0.613286  |          0.386714  |
   | Sex                                | female          |         -0.305838  |          0.305838  |
   | Pclass                             | 3               |          0.242448  |         -0.242448  |
   | Fare                               | 7.7375          |         -0.119392  |          0.119392  |
   | Siblings/Spouses Aboard            | 0               |         -0.0372811 |          0.0372811 |
   | ('Age', 'Parents/Children Aboard') | [28.0 0]        |          0.0122196 |         -0.0122196 |
   | PREDICTION                         |                 |          0.405443  |          0.594557  |
   +------------------------------------+-----------------+--------------------+--------------------+



Features
========
* Supports predictions explanations for classification and regression
* Easy to use API.
* Works with ``pandas`` and ``numpy``
* Support interactions between features


Installation
------------
Installation process is simple, just::

    $ pip install ibreakdown


Requirements
------------

* Python_ 3.6+
* numpy_

.. _Python: https://www.python.org
.. _numpy: http://www.numpy.org/
.. _iBreakDown: https://github.com/ModelOriented/iBreakDown
.. _Shapley: https://en.wikipedia.org/wiki/Shapley_value
