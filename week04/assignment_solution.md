# Exercise Sheet 2

## Exercise 2-1

a) 

ADALINE gradient descent based learnig rule (**Batch Gradient Descent**):

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \eta \sum_{i=1}^{N}{\boldsymbol{x}_i(y_i - \hat{y}_i)}
$$

Perceptron learning rule:

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \eta y_i \boldsymbol{x}_i
$$

b)

Sample-based rule for ADALINE (**Stochastic Gradient Descent** or **Delta Rule**):

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \eta \boldsymbol{x}_i(y_i - \hat{y}_i)
$$

c) 

SGD can be learned on the fly. The model can be update sample by sample. It's unessary to recompute the whole model, essentially better for large dataset.

d)

Striking difference: Objective function

> Be aware of: error / loss / cost and objective function

## Exercise 2-2

a)

- Overfitting: A trained model overadapt to a given dataset.
- Reason:
    + generalization explanation

    $$
    \text{Testing Error} \leq \text{Training Error} + \text{Model Complexity Penally}
    $$

    $$
    \text{Model Complexity} \approx O\left(\sqrt{\frac{\text{#features}}{\#samples}}\right)
    $$

    + traning & testing error relation during training produce
    + bias-variance trade-off: 

    $$
    \text{loss} = \text{bias} + \text{variance} + \text{irreducible error}
    $$


b) 

- Too much parameters
- Data is not enough
- Model complexity measurement



c)

- More data
- Regularization technique



## Exercise 2-3

a) 

- continues / differentiable almost everywhere

b) 

- not suitable

c)

- No. Consider RBF.
- Yes, of course.