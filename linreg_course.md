# Simple Linear Regression

## Definition
Simple linear regression (SLR) analysis is a statistical technique for investigating and modeling the relationship between two variables: Y and X variables. Customarily, *X* is called the independent (predictor or regressor) variable and *Y* is called the dependent (response) variable.

## Empirical Model
Assume that the expected value of Y is a linear function of X. The data points generally, but not exactly, fall along a straight line:

> ***$Y$* = $\beta$<small>0</small> + $\beta$<small>1</small>  *$X$* + *$\epsilon$*** 

where the intercept β<small>0</small> and the slope β<small>1</small> are unknown regression coefficients, and *$\epsilon$* = the random error term. 

A criterion for estimating the regression coefficients is called the method of ***least squares***.

From Equation 1, here to estimate β<small>0</small>:

> $\beta_0 = \bar{Y} + \beta_i \bar{X}$

where $\bar{Y}$ = $(\frac{1}{n})$ $\displaystyle\sum_{i=1}^n Y_i$ is the mean of Y,
$\bar{X}$ = $(\frac{1}{n})$ $\displaystyle\sum_{i=1}^n X_i$ is the mean of X.
and $i = 1, 2, ..., n$, $n$ is the number of observations.

To find $\beta$<small>1</small>: 

> $\beta$<small>1</small> = $\frac{\displaystyle\sum_{i=1}^n Y_i .X_i - \frac{(\displaystyle\sum_{i=1}^n Y_i) \times (\displaystyle\sum_{i=1}^n X_i)}{n}}{\displaystyle\sum_{i=1}^n X_i^2 - \frac{(\displaystyle\sum_{i=1}^n X_i)^2}{n}}$

The fitted or estimated regression line is therefore

> $\hat{Y} = \beta_i + \beta_1  X$

Note that each pair of observations satisfies the relationship

> $Y_i = \beta_i + \beta_1  X_i + \epsilon$

The difference between the observed value $Y_i$ and the corresponding fitted value $\hat{Y_i}$ is a residual. Mathematically the $i$-th residual is

> $e_i = Y_i - \hat{Y_i}$

## Representation
As an example of a problem, consider the data in Table 1, y is the purity of oxygen produced in a chemical distillation process, and x is the percentage of hydrocarbons present in the main condenser of the distillation unit. The 20 observations are plotted in Figure 1, called a ***scatter diagram***.

**Table 1. Oxygen and Hydrocarbon Levels**

|observation number | hydrocarbon level, x (%) | purity, y (%)|
|---- | ------------ | ------------|
|1    | 0.99 | 90.01 |
|2    | 1.02 | 89.05 |
|3    | 1.15 | 91.43 |
|4    | 1.29 | 93.74 |
|5    | 1.46 | 96.74 |
|6    | 1.36 | 94.45 |
|7    | 0.87 | 87.59 |
|8    | 1.23 | 91.77 |
|9    | 1.55 | 99.42 |
|10   | 1.40 | 93.65 |
|11   | 1.19 | 93.54 |
|12   | 1.15 | 92.52 |
|13   | 0.98 | 90.56 |
|14   | 1.01 | 89.54 |
|15   | 1.11 | 89.85 |
|16   | 1.20 | 90.39 |
|17   | 1.26 | 93.25 |
|18   | 1.32 | 93.41 |
|19   | 1.43 | 94.98 |
|20   | 0.95 | 87.33 |


The data points generally, but not exactly, fall along a straight line, see Figure 2.

![Scatter diagram of oxygen purity versus hydrocarbon level from Table 1](d://z/data_analysis/linear_regression/raw/fig1_1.jpg)

**Figure 1.** Scatter diagram of oxygen purity versus hydrocarbon level from Table 1.

![Straight-line relationship between oxygen purity y and  hydrocarbon level x](d://z/data_analysis/linear_regression/raw/fig_2.jpg)

**Figure 2.** Straight-line relationship between oxygen purity y and  hydrocarbon level x.


Objectives: (a) Data description, (b) Prediction and estimation, (c) Process control
