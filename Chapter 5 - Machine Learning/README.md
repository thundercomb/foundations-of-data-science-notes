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

### Question: What is a simple linear separator algorithm?

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

### Question: How does the Perceptron algorithm work?

**w** ⟵ 0  
**w** ⟵ **w** + **x<sub>i</sub><i>l<sub>i</sub></i>**     while **x<sub>i</sub><i>l<sub>i</sub></i>·w** ≤ 0 for a valid **x<sub>i</sub>**

Thus each new (<b>ŵ · x̂<sub>i</sub></b>)<i>l<sub>i</sub></i> will be increased by **x<sub>i</sub> · x<sub>i</sub>**<i>l<sub>i</sub></i> = |**x<sub>i</sub>**|<sup>2</sup>. The number of steps to find a solution **w** depends on the *margin of separation*.

The *margin of the separator* is defined as the minimum distance of any **x<sub>i</sub>** to the *linear separator* <b>w</b><sup>\*</sup> · <b>x</b> = 0. If <b>w</b><sup>\*</sup> is scaled such that (<b>w</b><sup>\*</sup> · <b>x<sub>i</sub></b>)<i>l<sub>i</sub></i> ≥ 1 then the margin of the separator will be at least 1/|<b>w</b><sup>\*</sup>|. 

Suppose all points lie inside a ball with radius *r*, then *r*|<b>w</b><sup>\*</sup>| is the ratio of the radius to the margin. The theorem demonstrates that the number of update steps is at most *r*<sup>2</sup>|<b>w</b><sup>\*</sup>|<sup>2</sup>.

**Perceptron theorem:**

<i>If there is a </i><b>w</b><sup>\*</sup><i> satisfying </i>(<b>w</b><sup>\*</sup> · <b>x<sub>i</sub></b>)<i>l<sub>i</sub></i> ≥ 1 <i>for all i, then the Perceptron Algorithm finds a solution </i><b>w</b> <i>with</i> (<b>w · x<sub>i</sub></b>)<i>l<sub>i</sub></i> > 0 <i>for all i in at most </i>*r*<sup>2</sup>|<b>w</b><sup>\*</sup>|<sup>2</sup> <i>updates where r</i> = max<i><sub>i</sub></i> |**x<sub>i</sub>**|.

The multiplication by <i>l<sub>i</sub></i> is a simple device to force the overall value to be positive. I.e., if the prediction is a negative (as it is hoped to be), then multiplying by the true negative label (-1) provides the positive outcome. If both are positive, it's the same. The assumption is that it is linearly separable, in which case the right <b>w</b> exists.

The update only happens when the predicted and true labels are not aligned, i.e. one is positive and the other negative. In that case <b>x<sub>i</sub></b><i><sup>T</sup></i><i>l<sub>i</sub></i><b>w</b> ≤ 0.

Note: Given **w** looks like a row vector, Blum et al's notation appears a little strange. The assumption is that both <b>w</b><sup><i>T</i></sup><b>w</b><sup>\*</sup> and |<b>w</b>|<sup>2</sup> are scalar quantities. |<b>w</b>|<sup>2</sup> is the square of the vector norm of **w**, which is the same as the dot product of **w** with itself. As for <b>w</b><sup><i>T</i></sup><b>w</b><sup>\*</sup>, it appears to be the equivalent of |<b>w</b>||<b>w</b><sup>*</sup>|, which is the product of two vector norms, and hence two scalars.

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

### Question: What is a positive semi-definite matrix?

Given:  
*M*: a symmetric *n* × *n* matrix  
*z*: a non-zero column vector of *n* real numbers

If *z*<sup>T</sup>*Mz* is strictly positive or zero (i.e. non-negative) it is called a positive semi-definite matrix.

Thus:

*M* positive semi-definite ⟷ *z*<sup>T</sup>*Mz* ≥ 0 for all *z* ∈ R<sup>n</sup>

Importantly, it implies that "the output always has a positive inner product with the input, as often observed in physical processes" (Wikipedia).

### Question: What about non-linearly separable data?

Linearly separable data can take advantage of the Perceptron Algorithm to find a solution. In the case of non-linearly separable data, a map to a higher dimensional space where it is linearly separable means that the Perceptron Algorithm can be applied to the mapped space instead.

A common method for doing this is the kernel method, which uses kernel functions. Kernels are typically real-valued functions. Kernel methods can operate in a high dimensional space.

**Kernel Lemma:**

<i>A matrix K is a kernel matrix, i.e., there is a function 𝜑 such that k<sub>ij</sub> = 𝜑</i>(<b>x<sub>i</sub></b>)<i><sup>T</sup>𝜑</i>(<b>x<sub>j</sub></b>)<i>, if and only if K is positive semi-definite</i>.

*K* can be expressed as *K = BB<sup>T</sup>* because it is positive semi-definite. If we define 𝜑</i>(<b>x<sub>i</sub></b>)<i><sup>T</sup></i> to be the *i<sup>th* row of *B*, then <i>k<sub>ij</sub> = 𝜑</i>(<b>x<sub>i</sub></b>)<i><sup>T</sup>𝜑</i>(<b>x<sub>j</sub></b>). Conversely, suppose there is an embedding 𝜑 such that <i>k<sub>ij</sub> = 𝜑</i>(<b>x<sub>i</sub></b>)<i><sup>T</sup>𝜑</i>(<b>x<sub>j</sub></b>), then by using 𝜑(<b>x<sub>i</sub></b>)<i><sup>T</sup></i> for the rows of the matrix *B*, matrix *K* will be *K = BB<sup>T</sup>*. Thus *K* is positive semi-definite.

As there are a number of legal kernel functions, one of the easiest ways of creating a new kernel function is to combine them.

**Kernel combination theorem:**

<i>Suppose k</i><sub>1</sub><i> and k</i><sub>2</sub><i> are kernel functions. Then

1. For any constant c </i>≥ 0<i>, ck</i><sub>1</sub><i> is a legal kernel. In fact, for any scalar function f, the function k</i><sub>3</sub>(**x, y**) = *f*(**x**)*f*(**y**)<i>k</i><sub>1</sub>(**x, y**)<i> is a legal kernel.
2. The sum k</i><sub>1</sub><i> + k</i><sub>2</sub><i> is a legal kernel
3. The product k</i><sub>1</sub><i>k</i><sub>2</sub><i> is a legal kernel</i>

By this token *k*(**x, y**) = (1 + **x**<i><sup>T</sup></i>**y**)<sup>k</sup> is legal because *k*<sub>1</sub>(**x, y**) = 1 is a legal kernel, and so is *k*<sub>2</sub>(**x, y**) = **x**<i><sup>T</sup></i>**y**. Add them together and multiply the sum by itself *k* times.

Examples of kernels:

Linear kernel: *k*(**x, y**) = **x**<i><sup>T</sup></i>**y**,&nbsp;&nbsp;&nbsp;&nbsp;**x, y** ∈ R<sup>d</sup>  
Polynomial kernel: *k*(**x, y**) = (**x**<i><sup>T</sup></i>**y** + *r*)<sup>n</sup>,&nbsp;&nbsp;&nbsp;&nbsp;**x, y** ∈ R<sup>d</sup>,&nbsp;&nbsp;*r* ≥ 0, *n* ≥ 1  
Gaussian kernel: *k*(**x, y**) = *e*<sup>-<i>c</i>|**x** - **y**|<sup>2</sup></sup>,&nbsp;&nbsp;&nbsp;&nbsp;**x, y** ∈ R<sup>d</sup>    

### Question: How does an algorithm generalise to new data?

Two conditions apply. Firstly, both the training data and the new data should be drawn from the same probability distribution. Secondly, the rule (eg. classification rule) should be simple.

**Formally**:

*X*: The instance space  
*D*: Probability distribution over *X*  
*S*: training set points drawn independently at random from *D*  

*Objective:* predict well on new points also drawn from *D*  

*c*<sup>\*</sup>: target concept - subset of *X* corresponding to the positive class for binary classification concept  
*h* : hypothesis  

*Objective:* *h* ⊆ *X* as close as possible to *c*<sup>\*</sup> (low true error)  

*err<sub>D</sub>(h)*: the *true error* of *h* - *err<sub>D</sub>(h)* = Prob(*h*∆*c*<sup>\*</sup>)  
*err<sub>S</sub>(h)*: the *training error* of *h* - *err<sub>S</sub>(h)* = |*S*∩(*h*Δ*c*<sup>\*</sup>)|/|S|  
*overfitting*: low training error yet high true error  
*H*: a collection of subsets (hypotheses or concepts) over *X* called the hypothesis class

For example, if *X* = R, then the class of *intervals* over *X* is the collection {[*a*, *b*]|a ≤ b}. The class of *linear separators* over R<i><sup>d</sup></i> is the collection {{**x** ∈ R<i><sup>d</sup></i>|**w**·**x** ≥ *t*}|**w** ∈ R<i><sup>d</sup></i>, *t* ∈ R}} ; i.e. the collection of all sets in R<i><sup>d</sup></i> that are linearly separable from their complement.

*Objective:* to find a hypothesis in H that corresponds highly with *c*<sup>\*</sup> over S, while the *training error* is close to *true error*  

To ensure the *training error* for all *h* ∈ H is close to the *true error*, the argument is that *S* should be large enough with respect to some property  of H.  

### Question: What generalisation guarantees do we have?

We have a hypothesis class H and many possible *h*/ We want to select the one where the generalisation error (the *true error*) is low.

The book offers two generalisation guarantees that help guard against overfitting. 

In the first, the case is addressed where a hypothesis exists where the training error is zero. In the second, the case is addressed where a hypothesis has training error greater than zero.

### Question: What is the PAC Learning Guarantee?

The PAC learning guarantee looks at the case where the training error for a hypothesis is zero.

Given a situation where ϵ is greater than zero, and the training set is large compared to <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{1}{\epsilon}\textrm{ln}(\mid{\textrm{H}}\mid)">, it is unlikely that any hypothesis *h* in H that has true error greater than ϵ, will have zero training error; i.e. if *h* agrees with the relevant target concept on training data, the true error will be low as well.

**PAC Learning Theorem:**

<i>Let H be an hypothesis class and let ϵ > 0, ẟ > 0. If a training set S of size

<p align="center">
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=n \ge \frac{1}{\epsilon}\left(\textrm{ln}\mid\textrm{H}\mid %2B\,\textrm{ln}(\frac{1}{\delta})\right)">
</p>

is drawn from distribution D, then with probability greater than or equal to 1 - ẟ every h in H with true error err<sub>D</sub>(h) ≥ ϵ has training error *err<sub>D</sub>(h)* ≥ 0. Equivalently, with probability greater than or equal to 1 - ẟ, every h ∈ H with training error zero has true error less than ϵ.</i>  

Another way of stating this is that, assuming the same distribution *D*, then with probability at least 1 − δ, a hypothesis *h* ∈ H may be found with a true error less than or equal to ϵ.

<i>h<sub>1</sub></i> to <i>h<sub>m</sub></i>: hypotheses in H with error greater than or equal to ϵ (undesirable outputs)  
*S*: sample of size *n*
<i>A<sub>i</sub></i>: the event that <i>h<sub>i</sub></i> has *err<sub>S</sub>(h)* = 0
For every <i>h<sub>i</sub></i>, *err<sub>D</sub>(h)* ≥ ϵ  
Thus:  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Prob(<i>A<sub>i</sub></i>) ≤ (1 - ϵ)<sup>n</sup>

In other words, the probability that the hypothesis will retain its validity throughout a sample *S* of size *n*, is (1 - ϵ)<i><sup>n</sup></i>

Given:  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (1 - ϵ) ≤ *e*<sup>-ϵ</sup>  

Then:  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Prob(<i>A<sub>i</sub></i>) ≤ |H|*e*<sup>-ϵ<i>n</i></sup>  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ≤ |H|*e*<sup>ln|H|-ln(1/ẟ)</sup>  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ≤ ẟ  

This guarantee is often called the Probability Approximately Correct (PAC) learning guarantee. It addresses the case where a hypothesis has zero training error. But what guarantees do we have if it doesn't.