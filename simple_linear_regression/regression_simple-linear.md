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

![**Figure 1.** Scatter diagram of Y versus X](D://z/data_analysis/linear_regression/raw/fig_4.jpg)
![**Figure 1.** Scatter diagram of Y versus X](https://github.com/agdhr/data_analysis/blob/main/linear_regression/fig_4.jpg)

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

> $\hat{\beta_1} = \frac{Sxy}{Sxx}$

where $S_{xx} = \displaystyle\sum_{i=1}^n x_i^2 - \frac{(\displaystyle\sum_{i=1}^n x_i)^2}{n}$
and
$S_{xy} = \displaystyle\sum_{i=1}^n y_i .x_i - \frac{(\displaystyle\sum_{i=1}^n y_i) \times (\displaystyle\sum_{i=1}^n x_i)}{n}$

Since the denominator is the corrected sum of squares of the $x_i$ and the numerator is the corrected sum of cross products of $x_i$ and $y_i$, we may write $Sxx$ and $Sxy$ in a more compact notation as:

> $S_{xx} = \displaystyle\sum_{i=1}^n (x_i -\bar{x})^2$

> $S_{xy} = \displaystyle\sum_{i=1}^n (y_i - \bar{y}) (x_i - \bar{x})$

The fitted or estimated regression line is therefore

> $\hat{y_i} = \hat{\beta_0} + \hat{\beta_1}  x_i$

## Step-by-step Procedures

**Step 1 - Compute the means $\bar{X}$ and $\bar{Y}$, the corrected sum of squares $\sum{X^2}$ and $\sum{Y^2}$, and the corrected sum of cross product $\sum{XY}$**

Use a spreadsheet 

| obs   | $x$   | $y$   | $x^2$ | $y^2$ | $xy$  | $(x_i - \bar{x})$ | $(y_i - \bar{y})$ | $(x_i - \bar{x}) (y_i - \bar{y})$|
|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ...   | ...   | ...   | ... | ... | ... | ... | ... | ... |
| ...   | ...   | ...   | ... | ... | ... | ... | ... | ... |
| ...   | ...   | ...   | ... | ... | ... | ... | ... | ... |
| n   | $\sum{x}$= ...   | $\sum{y}$=...   | $\sum{x^2}$=... | $\sum{y^2}$=... | $\sum{xy}$=... | $\sum({x_i - \bar{x})}$=... | $\sum({y_i - \bar{y})}$=...| $\sum{(x_i - \bar{x}) (y_i - \bar{y})}$ = ...|
|  | $\bar{x}$= ...   | $\bar{y}$=...   |  |  |  |  |  |  |  |  |

**Step 2 - Compute the estimates of the regression parameters: the intercept $\beta_0$ and the slope $\beta_1$**

$\hat{\beta_1} = \frac{S_{xy}}{S_{xx}} = \frac{\displaystyle\sum_{i=1}^n (y_i - \bar{y}) (x_i - \bar{x})}{\displaystyle\sum_{i=1}^n (x_i -\bar{x})^2}$

or

$\hat{\beta_1} = \frac{S_{xy}}{S_{xx}} = \frac{\displaystyle\sum_{i=1}^n y_i .x_i - \frac{(\displaystyle\sum_{i=1}^n y_i) \times (\displaystyle\sum_{i=1}^n x_i)}{n}}{\displaystyle\sum_{i=1}^n x_i^2 - \frac{(\displaystyle\sum_{i=1}^n x_i)^2}{n}}$

and

$\hat{\beta_0} =\bar{y} - \hat{\beta_1} \bar{x}$

**Step 3 - Plot the observed points and draw a graphical representation of the estimated regression equation**

The fitted or estimated regression line is obtained from

> $\hat{y_i} = \hat{\beta_0} + \hat{\beta_1}  x_i$

See figure 1-left for the sample plot

**Step 4 - Test the significance of the slope $\beta_1$**

The hypotheses:

* $H_0 :  \beta_1 = \beta_{1,0}$

* $H_0 :  \beta_1 \neq \beta_{1,0}$

where $\beta_{1,0}$ = the slope equals to a constant

The *t*-statistic:

$t_0 = \frac{\beta_1}{\sqrt{\frac{\hat{\sigma^2}}{S_{xx}}}}$

follows the t-distribution with $(n-2)$ degrees of freedom $\alpha$ under $H_0$. 

We would reject $H_0$ if

$|t_0| > t_{\frac{\alpha}{2},n-2}$

* Failure to reject $H_0$ if there is no linear relationship between X and Y (Figure 2-left)

*  Reject $H_0$ if there is a linear relationship between X and Y (Figure 2-middle)

**Step 5 - Test the hypotheses of the intercept $\beta_0 = \beta_{0,0}$** 

The hypotheses:

* $H_0 :  \beta_0 = \beta_{0,0}$

* $H_0 :  \beta_0 \neq \beta_{0,0}$

The *t*-statistic:

$t_0 = \frac{\beta_0}{\sqrt{\hat{\sigma^2} (\frac{1}{n} + \frac{\bar{x^2}}{S_{xx}})}}$

follows the t-distribution with $(n-2)$ degrees of freedom $\alpha$ under $H_0$. 

We would reject $H_0$ if

$|t_0| > t_{\frac{\alpha}{2},n-2}$

* Failure to reject $H_0$ if $\beta_0$ is significantly different from $\beta_{0,0}$

* Reject $H_0$ if  $H_0$ if $\beta_0$ is signficantly different from $\beta_{0,0}$

**Step 6 - Construct the 100(1-$\alpha$)% confidence interval (C.I.) for the intercept, the slope, the mean response, and the future prediction**

* **For the intercept $\beta_0$**

C.I. = $\beta_0 \pm t_{\frac{\alpha}{2},n-2} \sqrt{\hat{\sigma^2} (\frac{1}{n} + \frac{\bar{x^2}}{S_{xx}})}$ 

Therefore, $...... < \beta_0 < .....$

If the $100 (1-α )%$ C.I. does not include zero, so there is strong evidence (at $α$) that the intercept is not zero. 

* **For the slope $\beta_1$**

C.I. = $\beta_0 \pm t_{\frac{\alpha}{2},n-2} \sqrt{\frac{\hat{\sigma^2}}{S_{xx}}}$ 

Therefore, $...... < \beta_1 < .....$

If the $100 (1-α )%$ C.I. does not include zero, so there is strong evidence (at $α$) that the slope is not zero. 

* **For the mean response $\mu$**

$\mu_{Y|x_i} = \mu_{Y|x_i} \pm t_{\frac{\alpha}{2},n-2} \sqrt{\hat{\sigma^2} (\frac{1}{n} + \frac{(x_i - \bar{x^2})}{S_{xx}})}$ 

Therefore, $...... < \mu_{Y|x_i} < .....$

where $\mu_{Y|xi} =\beta_i + \beta_1 x_i$ is computed from the fitted regression model.

* **For the future prediction $\hat{y}$**

$\hat{y_i} = \hat{y_i} \pm t_{\frac{\alpha}{2},n-2} \sqrt{\hat{\sigma^2} (\frac{1}{n} + \frac{(x_i - \bar{x^2})}{S_{xx}})}$ 

Therefore, $...... < \hat{y_i} < .....$

where $\hat{y_i}$  is computed from $\hat{y_i} =\beta_i + \beta_1 x_i$ is computed from the fitted regression model.

**Step 7 - ANOVA for regression**

It is used to provide information about levels of variability within a regression model and form a basis for tests of significance.

> DATA = FIT + RESIDUAL

$\displaystyle\sum_{i=1}^n (y_i - \bar{y})^2 = \displaystyle\sum_{i=1}^n (\hat{y_i} - \bar{y})^2 = \displaystyle\sum_{i=1}^n (y_i - \hat{y})^2$

> $SS_T = SS_E + SS_R$

where $SS_R = \beta_1 S_{xy}$

The degree of freedom
> $df_T = df_E+df_R$

The F statistic:

|source of variation | sum of squares| degree of freedom | mean suares | $F_0$ |
|----- | -----| ----- |----- |-----|
|Residual | $SS_R$| n-2 | $MS_R$ | $MS_R/MS_E$|
|Error | $SS_E$ | 1 | $MS_E$ |  |
|Total | $SS_T$ | n-1 | |  |

The hypotheses:

* $H_0 :  \beta_0 = 0$

* $H_0 :  \beta_0 \neq0$

reject $H_0$ if 

$f_0 > f_{\alpha,1,n-2}$

The F-test is equivalent to t-he test. Hence, it will lead to the same conclusion.

![Figure 2. The situations where Ho is rejected (left) and is not rejected (middle); (right) Scatter diagram of X and Y with fitted regression line, 95% prediction limits (outer lines), and 95% confidence limits on μ](D://z/data_analysis/linear_regression/raw/fig_3.jpg)

![Figure 2. The situations where Ho is rejected (left) and is not rejected (middle); (right) Scatter diagram of X and Y with fitted regression line, 95% prediction limits (outer lines), and 95% confidence limits on μ](https://github.com/agdhr/data_analysis/blob/main/linear_regression/fig_3.jpg)

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

_

![Straight-line relationship between oxygen purity y and  hydrocarbon level x](d://z/data_analysis/linear_regression/raw/fig_2.jpg)

![Straight-line relationship between oxygen purity y and  hydrocarbon level x](https://github.com/agdhr/data_analysis/blob/main/linear_regression/fig_2.jpg)

## Reference

Montgomery, D. C., & Runger, G. C. (2014). Applied Statistics and Probability for Engineers, 6th edition. John Wiley & Sons, Inc.

Gomez, K. A. and Gomez, A. A. (1984). Statistical Procedures for Agricultural Research, 2nd edition.  John Wiley & Sons, Inc. 