# Exercise Sheet 3

## Exercise 3-1

a)

N/A

b)

$\sup{\hat{y}} = \sum_{h=0}^{M_\phi - 1}{w_h}$, $\inf{\hat{y}} = w_0$

c)

$\hat{y} = w_0 + 0.5 \sum_{h=0}^{M_\phi - 1}{w_h}$

$\hat{y} = w_0 + \frac{1}{1+\exp(1+x)}\sum_{h=0}^{M_\phi - 1}{w_h} = \text{offset} + \text{sigmoid_transformation} \times \text{weights_factor}$

## Exercise 3-2

Left:

0 0 0 0 1 1 1 1 0 0 0 0

0 0 1 1 0 0 -1 -1 0 0

0 0 -1 -1 0 0 1 1 0 0

0 1 0 -1 0

0 -1 0 1 0

0 -4 0

0.99 0.01



Right:

1 1 1 1 0 0 0 0 1 1 1 1

0 0 -1 -1 0 0 1 1 0 0

0 0 1 1 0 0 -1 -1 0 0

0 -1 0 1 0

0 1 0 -1 0

0 4 0

0.01 0.99