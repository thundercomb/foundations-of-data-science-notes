# Foundations of Data Science
## Chapter 5 : Machine Learning
### Question: What is a core problem in Machine Learning?

Learning a good classification from labelled data is a common and central problem that machine learning tries to deal with.

This problem has two components: the *instance space* (domain of interest) for example email messages, and the classification task.

The instance space can typically be assumed to be [0, 1]<sup>d</sup> or R<sup>d</sup>, with data having *d* boolean, categorical (including ordinal), or real-valued features.

The focus is on generalisation.

The chapter will focus on *binary classification* (most of the techniques apply equally to multi-way classification).

### Question: What is the general aim of solving the classification problem?

With reference to the generalisation principle, the idea is that we use training data that is representative of what future data might look like. We then strive to discover a simple enough "rule", for example a *linear separator*

This could imply, as an example, learning positive and negative weights while training on email messages, to learn which ones are spam and which are not.

In order to formulate this mathematically, we need to be precise about the meanings of:
- simple
- representative

### What is a simple linear separator algorithm?

The linear separator is a simple rule in *d*-dimensional space. "Does a weighted sum of feature values exceed a threshold?"

A neural network is a network of interconnected threshold gates. A threshold gate is sometimes called a perceptron.

Given n labelled examples <b>x<sub>1</sub></b>, <b>x<sub>2</sub></b>, ... ,<b>x<sub>n</sub></b> in *d*-dimensional space,
where each example has label +1 or -1,
our task is to find a *d*-dimensional vector **w**, if one exists, with a threshold *t* where:

<p align="center">
<b>w · x<sub>i</sub></b> > <i>t</i> for each <b>x<sub>i</sub></b> labelled +1  
</br>
<b>w · x<sub>i</sub></b> < <i>t</i> for each <b>x<sub>i</sub></b> labelled -1
</p>

*Linear separator:* a vector-threshold pair (**w**, **t**) that satisfies the inequalities.  
One might look for a general-purpose algorithm, but the *perceptron algorithm* is a simpler algorithm that permits solution of the above problem statement if there is a feasible **w** and a fair amount of leeway.

Modify by adding coordinates (allows separator to include the origin):

<b>x̂<sub>i</sub></b> = (<b>x<sub>i</sub></b>, 1)
</br>
<b>ŵ</b> = (<b>w</b>, <i>-t</i>)

<p align="center">
(<b>ŵ · x̂<sub>i</sub></b>)<i>l<sub>i</sub></i> > 0 &nbsp&nbsp&nbsp 1 ≤ <i>i</i> ≤ <i>n</i>
</p>

where *l* is the label -1 or +1

From here on we will simply assume:

<b>x<sub>i</sub></b> = <b>x̂<sub>i</sub></b>
</br>
<b>w</b> = <b>ŵ</b>

### How does the Perceptron algorithm work?

**w** ⟵ 0  
**w** ⟵ **w** + **x<sub>i</sub><i>l<sub>i</sub></i>**     while **x<sub>i</sub><i>l<sub>i</sub></i>·w** ≤ 0 for a valid **x<sub>i</sub>**

Thus each new (<b>ŵ · x̂<sub>i</sub></b>)<i>l<sub>i</sub></i> will be increased by **x<sub>i</sub> · x<sub>i</sub>**<i>l<sub>i</sub></i> = |**x<sub>i</sub>**|<sup>2</sup>. The number of steps to find a solution **w** depends on the *margin of separation*.

The *margin of the separator* is defined as the minimum distance of any **x<sub>i</sub>** to the *linear separator* <b>w</b><sup>\*</sup> · <b>x</b> = 0. If <b>w</b><sup>\*</sup> is scaled such that (<b>w</b><sup>\*</sup> · <b>x<sub>i</sub></b>)<i>l<sub>i</sub></i> ≥ 1 then the margin of the separator will be at least 1/|<b>w</b><sup>\*</sup>|. 

Suppose all points lie inside a ball with radius *r*, then *r*|<b>w</b><sup>\*</sup>| is the ratio of the radius to the margin. The theorem demonstrates that the number of update steps is at most *r*<sup>2</sup>|<b>w</b><sup>\*</sup>|<sup>2</sup>.

**Perceptron theorem:**

<i>If there is a </i><b>w</b><sup>\*</sup><i> satisfying </i>(<b>w</b><sup>\*</sup> · <b>x<sub>i</sub></b>)<i>l<sub>i</sub></i> ≥ 1 <i>for all i, then the Perceptron Algorithm finds a solution </i><b>w</b> <i>with</i> (<b>w · x<sub>i</sub></b>)<i>l<sub>i</sub></i> > 0 <i>for all i in at most </i>*r*<sup>2</sup>|<b>w</b><sup>\*</sup>|<sup>2</sup> <i>updates where r</i> = max<i><sub>i</sub></i> |**x<sub>i</sub>**|.

The multiplication by <i>l<sub>i</sub></i> is a simple device to force the overall value to be positive. I.e., if the prediction is a negative (as it is hoped to be), then multiplying by the true negative label (-1) provides the positive outcome. If both are positive, it's the same. The assumption is that it is linearly separable, in which case the right <b>w</b> exists.

The update only happens when the predicted and true labels are not aligned, i.e. one is positive and the other negative. In that case <b>x<sub>i</sub></b><i><sup>T</sup></i><i>l<sub>i</sub></i><b>w</b> ≤ 0.

Note: Given w looks like a row vector, Blum et al's notation appears a little strange. The assumption is that both <b>w</b><sup><i>T</i></sup><b>w</b><sup>\*</sup> and |<b>w</b>|<sup>2</sup> are scalar quantities. |<b>w</b>|<sup>2</sup> is the square of the vector norm of **w**, which is the same as the dot product of **w** with itself. As for <b>w</b><sup><i>T</i></sup><b>w</b><sup>\*</sup>, it appears to be the equivalent of |<b>w</b>||<b>w</b><sup>*</sup>|, which is the product of two vector norms, and hence two scalars.

Suppose there is a <b>w</b><sup>\*</sup> satisfying (<b>w</b><sup>\*</sup> · <b>x</b><sub>i</sub>)<i>l<sub>i</sub></i> ≥ 1 for all *i*. Each update (<b>w</b><sup>\*</sup> · <b>x</b><sub>i</sub>)<i>l<sub>i</sub></i> increases <b>w</b><sup><i>T</i></sup><b>w</b><sup>\*</sup> by at least 1, so that:

<p align="center">
(<b>w</b> + <b>x<sub>i</sub></b><i>l<sub>i</sub></i>)<sup><i>T</i></sup><b>w</b><sup>*</sup> = <b>w<i></b><sup>T</sup></i><b>w</b><sup>*</sup> + <b>x<sub>i</sub><i></b><sup>T</sup></i><i>l<sub>i</sub></i><b>w</b><sup>*</sup> ≥ <b>w<i></b><sup>T</sup><b></i>w</b><sup>*</sup> + 1
</p>

Each update increases |<b>w</b>|<sup>2</sup> by at most <i>r</i><sup>2</sup> (where |<b>w</b>|<sup>2</sup> = <b>w</b> · <b>w</b>, i.e. <b>w</b> matrix multiplied by itself to give a scalar):

<p align="center">
(<b>w</b> + <b>x<sub>i</sub></b><i>l<sub>i</sub></i>)<sup><i>T</i></sup>(<b>w</b> + <b>x<sub>i</sub></b><i>l<sub>i</sub></i>) = |<b>w</b>|<sup>2</sup> + 2<b>x<sub>i</sub></b><i></b><sup>T</sup></i><i>l<sub>i</sub></i><b>w</b> + |<b>x<sub>i</sub></b><i>l<sub>i</sub></i>|<sup>2</sup> ≤ |<b>w</b>|<sup>2</sup> + |<b>x<sub>i</sub></b>|<sup>2</sup> ≤ |<b>w</b>|<sup>2</sup> + <i>r</i><sup>2</sup>
</p>

Given <i>m</i> updates, <b>w</b><i><sup>T</sup></i><b>w</b><sup>\*</sup> ≥ *m* and |<b>w</b>|<sup>2</sup> ≤ <i>mr<sup>2</sup></i>.  
Equivalently, |<b>w</b>||<b>w</b><sup>*</sup>| ≥ <i>m</i> and |<b>w</b>| ≤ <i>r√m</i>.

With this information it can be shown that *m* ≤ <i>r<sup>2</sup></i>|<b>w</b><sup>\*</sup>|<sup>2</sup>