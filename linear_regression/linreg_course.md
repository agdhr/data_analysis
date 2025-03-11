# Simple Linear Regression

## Definition
Simple linear regression (SLR) analysis is a statistical technique for investigating and modeling the relationship between two variables: Y and X variables. Customarily, *X* is called the independent (predictor or regressor) variable and *Y* is called the dependent (response) variable.

## Empirical Model
Assume that the expected value of Y is a linear function of X. The data points generally, but not exactly, fall along a straight line (Figure 1-left):

> $Y = \beta_0 + \beta_1  X + \epsilon$ 

where the intercept β<small>0</small> and the slope β<small>1</small> are unknown regression coefficients, and *$\epsilon$* = the random error term. 


## Least-Squares Estimation
A criterion for estimating the regression coefficients is called the method of ***least squares***. 
The sum of the squares of the differences *$\epsilon$* between the observations $y_i$ and the straight line is a ***minimum***.

The difference between the observed value $y_i$ and the corresponding fitted value $\hat{y_i}$ is a residual. Mathematically the $i$-th residual is:

> $\epsilon_i = y_i - \hat{y_i}$


![Scatter diagram of Y versus X](https://github.com/agdhr/data_analysis/blob/main/linear_regression/fig_4.jpg)

![Scatter diagram of Y versus X](D://z/data_analysis/linear_regression/raw/fig_4.jpg)

Sum of the squares of the deviations of the observations (*$\epsilon_i$*):
$S = \displaystyle\sum_{i=1}^n \epsilon_i^2 = \displaystyle\sum_{i=1}^n (y_i - \hat{\beta_0} - \hat{\beta_1} . x_i)^2$ 

with $i = 1, 2, ..., n$, with $n$ is the number of observations.

$\hat{\beta_0}$ and $\hat{\beta_1}$ must satisfy:

$\frac{\partial S}{\partial \beta_0}|$ <small>$\hat{\beta_0}$</small>, <small>$\hat{\beta_1}$</small> = $-2 \displaystyle\sum_{i=1}^n (y_i - \hat{\beta_0} - \hat{\beta_1} . x_i) = 0$

$\frac{\partial S}{\partial \beta_1}|$ <small>$\hat{\beta_0}$</small>, <small>$\hat{\beta_1}$</small> = $-2 \displaystyle\sum_{i=1}^n (y_i - \hat{\beta_0} - \hat{\beta_1} . x_i) x_i = 0$

Simplifying these two equations yields:

$n \hat{\beta_0} + \hat{\beta_1} \displaystyle\sum_{i=1}^n x_i = \displaystyle\sum_{i=1}^n y_i$

$\hat{\beta_0} \displaystyle\sum_{i=1}^n x_i + \hat{\beta_1} \displaystyle\sum_{i=1}^n x_i^2 = \displaystyle\sum_{i=1}^n x_i y_i$

These are called the least-squares normal equations. The solution to estimate $\beta_0$:

> $\hat{\beta_0} = \bar{y} + \hat{\beta_i} \bar{x}$

where $\bar{y}$ = $(\frac{1}{n})$ $\displaystyle\sum_{i=1}^n y_i$ and
$\bar{x}$ = $(\frac{1}{n})$ $\displaystyle\sum_{i=1}^n x_i$ are the mean of X and Y, respectively.

To find $\hat{\beta_1}$: 

> $\beta_1 = \frac{Sxy}{Sxx}$

where $Sxx = \displaystyle\sum_{i=1}^n x_i^2 - \frac{(\displaystyle\sum_{i=1}^n x_i)^2}{n}$
and
$Sxy = \displaystyle\sum_{i=1}^n y_i .x_i - \frac{(\displaystyle\sum_{i=1}^n y_i) \times (\displaystyle\sum_{i=1}^n x_i)}{n}$

The fitted or estimated regression line is therefore

> $\hat{y} = \hat{\beta_0} + \hat{\beta_1}  x$

## Sample problem
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



**Figure 1.** Scatter diagram of oxygen purity versus hydrocarbon level from Table 1.

![Straight-line relationship between oxygen purity y and  hydrocarbon level x](d://z/data_analysis/linear_regression/raw/fig_2.jpg)

**Figure 2.** Straight-line relationship between oxygen purity y and  hydrocarbon level x.


Objectives: (a) Data description, (b) Prediction and estimation, (c) Process control
