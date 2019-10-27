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
