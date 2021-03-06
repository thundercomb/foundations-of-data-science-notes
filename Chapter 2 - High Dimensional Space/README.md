# Foundations of Data Science
## Chapter 2 : High-Dimensional Space
### Question: What is a "Zero mean, unit variance Gaussian”?

*Gaussian distribution*

[https://www.inf.ed.ac.uk/teaching/courses/mlpr/2017/notes/w2b_univariate_gaussian.pdf](https://www.inf.ed.ac.uk/teaching/courses/mlpr/2017/notes/w2b_univariate_gaussian.pdf)

Math notation:</br>

<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=X \sim N( \mu, \sigma )"></p>  

<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=X \sim N( 0, 1 )"></p>


CS notation: <p align="center">x = randn( )</p>

Function: 

<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=f(x)={\frac {1}{\sqrt {2\pi }}}%20%20e^{-\frac{x^{2}}{2}}"></p>

a.k.a. Probability Density Function (PDF) for Standard Normal Distribution

The *inflection point* of the Gaussian (normal) curve is one standard deviation below the mean, and one standard deviation above the mean.

Let there be a curve with mean *μ* and standard deviation *σ*
Then inflection points occur where *x* = *μ* ± *σ*
 
 
*μ* : Mu  
*σ* : lower case sigma  

[https://www.thoughtco.com/inflection-points-of-a-normal-distribution-3126446](https://www.thoughtco.com/inflection-points-of-a-normal-distribution-3126446)

One property of the inflection point is that the second derivative of the function will be zero (however, not all points where the 2nd derivative of the function is zero, will be an inflection point).

So for a graph *y = f( x )*  
Given an inflection point at *x = a*  
Then *f’’( a )* = 0  

This can be proven using the Probability Density Function, as illustrated in the reference document above. The PDF is generally used in the context of continuous (rather than discrete) random variables. It can be used to specify the probability that a random variable falls *within a particular range of values*, as opposed to taking on a particular value (at a particular value the probability is zero).

See [https://en.wikipedia.org/wiki/Probability_density_function](https://en.wikipedia.org/wiki/Probability_density_function)

*Variance* (for continuous variables) can be expressed as:

<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sigma^2=\sum \frac{(X_i - X)^2}{N}"></p>  

Where:  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sigma^2"> = variance  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=(X_i - X)^2"> = (Individual Value – Mean)²  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sum">   = Summation of function associated with it  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math={N}"> = Total number of data points in our dataset  

This can also be expressed as: 
<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sigma^2=\sum \frac{(x_i - \mu)^2}{N}"></p>  

This is relevant in the context of the Gaussian curve, however it is worth keeping in mind that the situation would be different in the case of discrete variables.

[http://zerosnones.net/variance-limitation-properties-applications/](http://zerosnones.net/variance-limitation-properties-applications/)

"*Unit variance* means that the standard deviation of a sample as well as the *variance* will tend towards 1 as the sample size tends towards infinity." This is typically referred to as the *standard normal distribution*.

For a sample of a million random values:

```
import numpy as np
from matplotlib import pyplot as plt
N = 1000000
xx = np.random.randn(N)
hist_stuff = plt.hist(xx, bins=100)
plt.show()
```

![](Images/image_13.png)

*Variance and Standard Deviation*

[https://www.quora.com/What-are-zero-mean-unit-variance-Gaussian-random-numbers](https://www.quora.com/What-are-zero-mean-unit-variance-Gaussian-random-numbers)

*Variance* and *standard deviation* have a close relationship. The *standard deviation* is simply the square root of the *variance*.

We can see this quite easily:
```
import numpy as np

a = [11,9,5,13,18,6,9,12,10,7]

mean = np.sum(a) / len(a)
sqr_dev = []
for i in a:
    sqr_dev.append(np.square(abs(mean - i)))

variance = np.sum(sqr_dev) / len(a)
std_dev = np.sqrt(variance)

print(f"mean: {mean}")
print(f"original: {a}")
print(f"squared deviations: {sqr_dev}")
print(f"variance: {variance}")
print(f"standard deviation: {std_dev}")
print(f"np variance: {np.var(a)}")
print(f"np standard deviation: {np.std(a)}")
```

```
mean: 10.0
original: [11, 9, 5, 13, 18, 6, 9, 12, 10, 7]
squared deviations: [1.0, 1.0, 25.0, 9.0, 64.0, 16.0, 1.0, 4.0, 0.0, 9.0]
variance: 13.0
standard deviation: 3.605551275463989
np variance: 13.0
np standard deviation: 3.605551275463989
```

To conclude, a “zero mean, unit variance Gaussian” is a Gaussian distribution with *µ* =0, *σ =* 1, also called a Standard Normal Distribution.

Note that the Standard Normal Distribution is simply one kind of Normal Distribution. More generally, "the *normal distribution* is a probability function that describes how the values of a variable are *distributed* ” (Wikipedia). The mean and variance could have different values, for example it could have mean 4 and variance 2, as in the following example:

```
import numpy as np
from matplotlib import pyplot as plt

x = 4 + (np.random.randn(1000000) * np.sqrt(2))

print(x.mean()
print(x.var())
print(x.std())

4.000612096576726
2.0002004340052753
1.4142844247198918

h = plt.hist(x, bins=100)
plt.show()
```

![](Images/image_10.png)

So to return to our initial function:

<p align="center">
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=f(x)={\frac {1}{\sqrt {2\pi }}}%20%20e^{-\frac{x^{2}}{2}}">
</p>

"The factor <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac {1}{\sqrt {2\pi }}">  in this expression ensures that the total area under the curve *f* ( *x* ) is equal to one. The factor <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac {1}{2}"> in the exponent ensures that the distribution has unit variance (i.e., the variance is equal to one), and therefore also unit standard deviation. This function is symmetric around *x* = 0, where it attains its maximum value <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac {1}{\sqrt {2\pi }}"> and has [inflection points]() at *x* = +1 and *x* = −1.” - Wikipedia

### Question: How do you shift a distribution to zero mean?
 
 
**Answer:** Subtract the mean from (each value in) the distribution, and divide by the variance.

```
import numpy as np
from matplotlib import pyplot as plt

x = 10 + (np.random.randn(1000000))
normalised_x = (x - x.mean()) / x.std()

h = plt.hist(x, bins=100)
plt.show()
h = plt.hist(normalised_x, bins=100)
plt.show()
```

![](Images/image_7.png)

![](Images/image_18.png)

 
 
Given that in the above example the mean is ten, which we added to the dataset, and the variance should be close to 1 (unit variance), the formula is in effect simply reversing that operation: subtracting the mean, and dividing by the variance (≈1).

### Question: What is the Empirical Rule?

It is also known as the *68–95–99.7 rule* .  

It implies that 68.27%, 95.45% and 99.73% of the values lie within one, two and three standard deviations of the mean.

Mathematically it can be expressed as follows:

Pr( *µ* - 1σ ≤ *X* ≤ *µ* + 1σ ) ≈ 0.6827  
Pr( *µ* - 2σ ≤ *X* ≤ *µ* + 2σ ) ≈ 0.9545  
Pr( *µ* - 3σ ≤ *X* ≤ *µ* + 3σ ) ≈ 0.9973  

The *three-sigma rule* states that even in the case of non-normally distributed variables, 88.8% of cases will still fall within the three sigma intervals (three standard deviations from the mean). This follows from *Chebyshev’s inequality*, which will be looked at later.

The *three-sigma rule of thumb* is the observation that, for many cases (eg. in social sciences). most cases fall within three standard deviations of the mean. 

*Confidence intervals* can be calculated to provide a range of plausible values for the unknown parameter in question. The interval has an associated *confidence level.*
 
 
[https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)

The Gaussian has an important property in high dimensions. If we were to generate *n* points in *d* -dimensions, and each coordinate is a zero mean, unit variance Gaussian, and *d* is sufficiently large, then the *distance* between every pair of points will be almost exactly the same, with high probability.
 
 
### Question: What is a “Unit ball”?
 
 
A *unit circle* is the set of points of distance one from a fixed central point in two dimensions, in other words a circle with radius one. A *unit sphere* is the set of points in three dimensions of distance 1 from a fixed central point, in other words a sphere with radius one. The unit sphere is also known as a *unit ball*.

Any sphere can be transformed to a unit sphere through a combination of *translation* and *scaling* (Euclidian geometric transformations).

[https://en.wikipedia.org/wiki/Unit_sphere](https://en.wikipedia.org/wiki/Unit_sphere)

The unit ball has important properties in higher dimensions.

The *volume* of a high-dimensional unit ball is concentrated near its surface and also at its equator. One consequence of this is that the set of all points *x* such that | *x* | ≤ 1 goes to zero as *d* goes to infinity.

### Question: What is the Expected Value?
 
Intuitively the expected value is simply the value that can be expected following some kind of action or event. In statistics and probability, "the *expected value* is calculated by multiplying each of the possible outcomes by the likelihood each outcome will occur and then summing all of those *values*.” - [Investopedia](https://www.investopedia.com/terms/e/expected-value.asp)

It also has a connection to the Central Limit Theorem in that the “expected value is a measure of *central tendency*; a value for which the results will tend to. When a probability distribution is normal, a plurality of the outcomes will be close to the expected value.” - [Brilliant](https://brilliant.org/wiki/expected-value/)

The simple version of the formula is for binomial events, which simply states that you multiply the probability with the number of times that the event occurs:

<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=E(X) = P(x) X"></p>  

[https://www.youtube.com/watch?v=lxYBCrrhLW0](https://www.youtube.com/watch?v=lxYBCrrhLW0)

However, this is often too simple, so we need a way to represent multiple types of events. The formula in this case looks as follows:

<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=E(X) = \Sigma{X} \cdot P(X)"></p>  

[https://www.youtube.com/watch?v=_eIZKor-h48](https://www.youtube.com/watch?v=_eIZKor-h48)

These formulas are for discrete random variables, in other words where there are a countable number of possible values (even it is countably infinite). Each of these values have a probability between 0 and 1, and the sum of all the probabilities is equal to 1.

There is also the continuous case, and the general case. The differences between these types of random variables require a more rigorous exploration than is done in this section.

### Question: What is the Law of Large Numbers?

Informally, assuming we have a fair way of sampling, it simply means that the probability of reaching our expected value approaches 1 as the number of samples increases to infinity. A different way of putting is that the mean of the sample will approach the true mean of the population as *n* ⟶ ∞.

A simple example is the tossing of a coin. As the number of coin tosses *n* increases, the likelihood of us having tossed 50% heads and 50% tails tends towards 1. 

What it *does not mean* is that if we have a streak of heads that the probability of throwing tails suddenly increases. This is called the *Gambler’s Fallacy*. That is because the probability has no historical dependency, in other words it is *statistically independent*.    
 
[https://en.wikipedia.org/wiki/Gambler's_fallacy](https://en.wikipedia.org/wiki/Gambler's_fallacy)

There is an important distinction between the Weak and the Strong Law of Large Numbers. The Weak Law essentially says that there is no guarantee that all realised values of a random variable will fall within a given interval, in other words that the probability will be 1, but nevertheless there is a “high probability”. In the case of the Strong Law, however, we *can* say with probability 1 that the realised values will fall within the given interval because we are able to say that the values converge *almost surely*. 
 
[https://www.youtube.com/watch?v=Bn0wWZENeQI](https://www.youtube.com/watch?v=Bn0wWZENeQI)

*Weak Law:*

<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\lim _{n\to \infty }\,P\left(\mid{\bar{x}} - \mu \mid \ge \varepsilon \,\right)=0"></p>  

*Strong Law:*

<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=P\left(\lim _{n\to \infty }\,{\bar{x}}=E(x) \right)=1"></p>  

[https://terrytao.wordpress.com/2008/06/18/the-strong-law-of-large-numbers/](https://terrytao.wordpress.com/2008/06/18/the-strong-law-of-large-numbers/)

It is important to note that for the Law of Large Numbers to apply, the size of the population does not matter. The number *n* that should be sampled out of a population so that there is at most a chance 𝛿 that the estimate is off by more than *ϵ* depends only on 𝛿 and *ϵ* and not on the overall population size.

### Question: How do you Prove the Law of Large Numbers?
 
 
You can prove it by using Chebyshev’s inequality, which in turn relies on Markov’s inequality.

Proof of Markov’s inequality: [https://www.youtube.com/watch?v=sp9RF0zH-SU](https://www.youtube.com/watch?v=sp9RF0zH-SU)

Proof of Chebyshev’s inequality: [https://www.youtube.com/watch?v=h0YH79kLuOA](https://www.youtube.com/watch?v=h0YH79kLuOA)

Proof of the Law of Large Numbers: [https://www.youtube.com/watch?v=4QAeBJVn9WI](https://www.youtube.com/watch?v=4QAeBJVn9WI)

Courtesy of Ben Lambert’s excellent Youtube channel: [https://www.youtube.com/channel/UC3tFZR3eL1bDY8CqZDOQh-w](https://www.youtube.com/channel/UC3tFZR3eL1bDY8CqZDOQh-w)

### Question: What is the Central Limit Theorem?

The Central Limit Theorem is concerned with the sampling distribution of the mean.

"The central limit theorem states that if you have a population with mean *μ* and standard deviation *σ* and take sufficiently large random samples from the population with replacement, then the distribution of the sample means will be approximately normally distributed. This will hold true regardless of whether the source population is normal or skewed, provided the sample size is sufficiently large (usually n _>_ 30). If the population is normal, then the theorem holds true even for samples smaller than 30."

[http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Probability/BS704_Probability12.html](http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Probability/BS704_Probability12.html)

Suppose:  
The mean of the sample is *x̅*  
The size of the sample is *n*  
The standard deviation of the sample is *s*  
Then there are 4 aspects:  
1. the sampling distribution of the mean will be less spread than the values in the population
2. the sampling distribution will be well modelled by a normal distribution
3. the spread of the sampling distribution is related to the spread of the population values
4. bigger samples lead to a smaller spread in the sampling distribution

### Question: How does the Law of Large Numbers reflect in high dimensions?

"The square of the distance between two points *y* and *z* can be viewed as the sum of *d* independent samples of a random variable *x* that is the *squared difference of two Gaussians*.”

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mid \textbf{y} - \textbf{z} \mid^2 = \sum_{i=1}^d%20( y_i - z_i )^2">


Based on the Law of Large Numbers we can say that, with high probability, the sum is close to the sum’s expectation.

### Question: What is the difference between the Probability Density Function and Probability Mass Function?
 
 
Probability distribution can occur in two ways, depending on the characteristics of the random variable. If the random variable only has discrete values, the probability distribution is a Probability Mass Function (PMF) and the solution can be calculated as a weighted sum. If the random variable has continuous values, the probability is a Probability Density Function (PDF) and the solution can be calculated using an integral.
 
 
**PMF** - Probability mass function relates to the use of discrete random variables

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=P(X(x_i)) = P(X = x_i), \mathrm{for}\hspace{1mm} i = 1, 2, 3, \ldots,"></div>  

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\Sigma P(X(x_i)) = 1"></div>  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=P(x_i) > 0"></div>  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=P(x) = 0 \hspace{1mm}\mathrm{for}\hspace{1mm}\mathrm{all}\hspace{1mm}\mathrm{other} x"></div>  

**PDF** - Probability density function relates to the use of continuous random variables

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=fX(x) = dFX(x)dx = F'X(x), \mathrm{if} FX(x) \hspace{1mm}\mathrm{is}\hspace{1mm}\mathrm{differentiable}\hspace{1mm}\mathrm{at}\hspace{1mm} x"></div>  

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=P(-\infty \le X \le \infty) = \int_{\small -\infty}^{\small \infty} fX (x) dx"></div>  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=P( a \le X \le b ) = \int_{a}^{b} fX (x) dx"></div>  

 
A PDF in fact *describes* a continuous random variable. I.e. if a random variable takes values on a continuous set, that by itself isn’t enough to make it a continuous random variable.

The probability that X is any *particular* point is in fact 0 (the integral where *a* is equal to *b* is 0). A side effect of this is that the probability of a closed and open interval (one that does, and one that doesn’t include the endpoints) is identical.

*Note:* it is easy to confuse Probability with the Probability Density Function, however they are not the same. Probability density functions are not probabilities, but if *f(x)* is a probability density function, then <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=P = \int_{\small x_{\tiny 0}}^{\small x_{\tiny 1}}f(x)dx">  is a probability and thus <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\int_{\small x_{\tiny 0}}^{\small x_{\tiny 1}}f(x)dx \le 1\,\,\text{for all}\,x_0, x_1 (x_0 \le x_1)">. 

[https://math.stackexchange.com/questions/1720053/how-can-a-probability-density-function-pdf-be-greater-than-1](https://math.stackexchange.com/questions/1720053/how-can-a-probability-density-function-pdf-be-greater-than-1)


The general form of the PDF for the normal distribution is:

<p align="center">
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=f(x%3B\mu ,\sigma ^{2})={\frac {1}{\sigma {\sqrt {2\pi }}}}\hspace{1mm}e^{-{\frac {1}{2}}\left({\frac {x-\mu }{\sigma }}\right)^{2}}">
</p>

*Note: ‘ x; µ, σ ‘* implies that *x* depends on the other two parameters.

In the standard normal distribution *µ* = 0 and *σ* = 1, so that gives the simpler version:

<p align="center">
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=f(x)={\frac {1}{\sqrt {2\pi }}}\hspace{1mm}e^{-\frac{x^{2}}{2}}">
</p>

[https://www.youtube.com/watch?v=8QFpZ3FndBc](https://www.youtube.com/watch?v=8QFpZ3FndBc)
 
 
To prove that there is a vanishingly small probability that a randomly generated point **z** in *d*-dimensions would lie in the unit ball, Blum et al use a PDF with variance set at <img style="float: right;" src="https://render.githubusercontent.com/render/math?math={\frac {1}{2\pi }}"> so that the Gaussian probability density equals one at the origin (p. 7). This relationship is explained over here:

[https://books.google.co.uk/books?id=AKuMj4PN_EMC&pg=PA131&lpg=PA131&dq=probability+%22density+at+the+origin%22&source=bl&ots=EMqkf67xBd&sig=ACfU3U0TRJIjhaLxZlpSY2eEOxz5au52JA&hl=en&sa=X&ved=2ahUKEwiQ4I64k4bqAhUTrHEKHakYBM0Q6AEwB3oECAgQAQ#v=onepage&q=probability%20%22density%20at%20the%20origin%22&f=false](https://books.google.co.uk/books?id=AKuMj4PN_EMC&amp;pg=PA131&amp;lpg=PA131&amp;dq=probability+%22density+at+the+origin%22&amp;source=bl&amp;ots=EMqkf67xBd&amp;sig=ACfU3U0TRJIjhaLxZlpSY2eEOxz5au52JA&amp;hl=en&amp;sa=X&amp;ved=2ahUKEwiQ4I64k4bqAhUTrHEKHakYBM0Q6AEwB3oECAgQAQ#v=onepage&amp;q=probability%20%22density%20at%20the%20origin%22&amp;f=false)

So, given the formula:

<p align="center">
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=P( \textbf{x} = 0 ) = \frac{1}{(2\pi\sigma^2)^{N/2}}">
</div>

if we set the variance to <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{1}{2\pi}"> then the probability at *x* = 0 becomes 1 because the function becomes <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=1^{N/2}"> and 1 to the power of anything remains 1.

The argument goes:
* there is vanishingly small probability that a random point **z** in *d*-dimensions would lie in the unit ball
* this implies the integral of the probability density over the unit ball is vanishingly small
* and because the probability density in the unit ball is bounded below by a constant ...
* the unit ball must have vanishingly small volume

### Question: What is a tail bound and how do we calculate its probability?
 
 
A tail bound is the bound probability of rare events. There are many ways to calculate tail bound probabilities. There are a number of famous theorems that can calculate bounds to various levels of precision, depending on the type of information available.

These include the Markov, Chebyshev and Chernoff inequalities, as well as others like the Higher Moments, Gaussian Annulus, and Power Law theorems.

[https://courses.cs.washington.edu/courses/cse312/11au/slides/09tails.pdf](https://courses.cs.washington.edu/courses/cse312/11au/slides/09tails.pdf)

There is a more general version called the *Master Tail Bounds Theorem* from which some of these theorems can be derived.

In summary, the theorem says that given:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=x = x_1 %2B x_2 %2B \ldots %2B x_n">   , and <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=x_i"> is iid  
Variance ≤ <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sigma^2">  
*µ* = 0  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=0 \le a \le \sqrt{2n\sigma^2}">  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mid{E(x^s_i)}\mid \le \sigma^2s! \hspace{1cm}  \mathrm{for}\hspace{1mm}s = 3, 4\hspace{1mm}\ldots\hspace{1mm}\lfloor(a^2/4n\sigma^2)\rfloor">  

Then:
<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=P( \mid{x}\mid \ge a ) \le 3e^{\frac{-a^2}{12n\sigma^2}}"></p>  

It gives a much stronger bound with respect to *a* than Markov and Chebyshev, for example. Here we have an exponential drop off with respect to *a*, whereas Markov gives us <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{1}{a}"> and Chebyshev gives us <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{1}{a^2}">

[https://medium.com/jun-devpblog/data-science-1-expectation-variance-law-of-large-numbers-2ff49caf8b7d](https://medium.com/jun-devpblog/data-science-1-expectation-variance-law-of-large-numbers-2ff49caf8b7d)

### Question: How is volume distributed in higher dimensions?
 

Volume in higher dimensions is distributed near the surface.

Consider an object *A* in *R*<sup>*d*</sup> shrunk by a small amount *𝛜*. The new object is represented by:

( 1 - *𝛜* ) *A* = { ( 1 - *𝛜* ) *x* | *x* ∈ *A* }

This means that:

volume( ( 1 - *𝛜* ) *A* ) = ( 1 - *𝛜* ) <sup>*d*</sup> volume( *A* )

Because 1 - *x* ≤ *e*<sup>*-x*</sup>, for any given object *A* in *R*<sup>*d*</sup> we get:

volume( ( 1 - *𝛜* ) *A* ) / volume( *A* ) = ( 1 - *𝛜* ) <sup>*d*</sup> ≤ *e*<sup>-*𝛜 d*</sup>
 
 
If we fix *𝛜* and *d* ⟶ ∞ then the above approaches 0. Thus the volume of *A* is in the portion not represented by ( 1 - *𝛜* ) *A*.

Given a unit ball *S* in *d*-dimensions the implication of the above is that at least 1 - *e*<sup>-*𝛜 d*</sup> of the volume of *S* is concentrated in a *d*-dimensional annulus of width *𝛜* at the perimeter. We can see that *e*<sup>-*𝛜 d*</sup> decreases rapidly as *d* increases, eg. for 𝛜 = 0.1, *d* = 3, *e*<sup>-*𝛜 d*</sup> = 0.741, and for 𝛜 = 0.1, d = 100, *e*<sup>-*𝛜 d*</sup> = 0.000045. So 1 - *e*<sup>-*𝛜 d*</sup> ⟶ 1 as *d* ⟶ ∞.

Another way of looking at it is to say that the unit ball’s volume is concentrated in a *d*-dimensional annulus of width <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=O\frac{1}{d}"> or <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=1 - \frac{1}{d}"> near the perimeter. Given a ball of radius *r*, the width where the volume is concentrated is <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=O(\frac{r}{d})">.

Q: Other than the intuition that large *d* accounts for the relationship between volume and the width of the annulus, how is the leap made, exactly, to the annulus’ width <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=O(\frac{1}{d})">?

### Question: How do you calculate the volume of the unit ball in d dimensions?

As *d* ⟶ ∞ , the volume of the ball goes to zero. In the first few dimensions the volume goes up, up to d = 7, and then it goes down.

The volume can be calculated using integration. There is more than one way. Integration in Cartesian coordinates have complicated integral limits, so polar coordinates are preferred.

Let *V*( *d* ) is the volume of the unit ball, and *A*( *d* ) is the surface. The proof starts with:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=V(d) = \int_{S^d}\int_{r=0}^{1} r^{d-1}\hspace{1mm}drd\Omega"></div>  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=V(d) = \int_{S^d}\hspace{1mm}d\Omega\int_{r=0}^{1} r^{d-1}\hspace{1mm}dr"></div>  
 <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=V(d) = \frac{1}{d}\int_{S^d}\hspace{1mm}d\Omega"></div>  
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=V(d) = \frac{A(d)}{d}">  
 
 
But this stops integration at the surface of the sphere. To allow it to go all the way to infinity, involve an exponential in a function called *I*( *d* ). *I*( *d* ) can then be calculated in both Cartesian and polar coordinates, which yields:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=I(d) = \pi^{\frac{d}{2}}"></div>  

and:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=I(d) = A(d)\frac{1}{2}\Gamma(\frac{d}{2})"></div>  

which together gives:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=A(d) = \frac{\pi^{\frac{d}{2}}}{\frac{1}{2}\Gamma(\frac{d}{2})}"></div>  

That produces the lemma for the surface area *A*( *d* ) and the volume *V*( *d* ):

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=A(d) = \frac{2\pi^{\frac{d}{2}}}{\Gamma(\frac{d}{2})}"></div>  

and

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=V(d) = \frac{2\pi^{\frac{d}{2}}}{d\Gamma(\frac{d}{2})}">  

Since <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\pi^{\frac{\small d}{\small 2}}"> is an exponential in <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{d}{2}"> and <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\Gamma(\frac{d}{2})"> grows as the factorial of <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{d}{2}">, the <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\lim_{d \to +\infty} V(d) = 0"> (as claimed at the start).

An intuitive explanation of the formula is available courtesy of Zach Star:

[https://www.youtube.com/watch?v=mXp1VgFWbKc](https://www.youtube.com/watch?v=mXp1VgFWbKc)

Note that the above steps are an oversimplification of the proof. For a more general look at a proof for the formula of the volume of a ball in higher dimensions, see:

[https://www.youtube.com/watch?v=XLq-cjwvS3M](https://www.youtube.com/watch?v=XLq-cjwvS3M)

Another example:

[https://www.usna.edu/Users/physics/mungan/_files/documents/Scholarship/HypersphereVolume.pdf](https://www.usna.edu/Users/physics/mungan/_files/documents/Scholarship/HypersphereVolume.pdf)
 
 
### Question: How is the volume of the unit ball distributed in higher dimensions?
 
 
The claim is that in high dimensions most of a unit ball's volume is concentrated near its "equator”. This is tricky to understand and not easy to visualise, partly because the 3-dimensional version seems to be atypical. One way of thinking about is that, in actual fact, the volume is distributed evenly and that one lap around the ‘equator’ passes by an increasingly larger amount of its interior, and therefore volume, as *d* ⟶ ∞. In terms of a visualisation it helps to keep in mind that a unit ball becomes ’spiky’, however with some caveats as explained by Colin Wright:

[http://www.penzba.co.uk/cgi-bin/PvsNP.py?SpikeySpheres](http://www.penzba.co.uk/cgi-bin/PvsNP.py?SpikeySpheres)

[https://news.ycombinator.com/item?id=3995615](https://news.ycombinator.com/item?id=3995615)

Blum et al puts it as follows: “most of the volume of the unit ball lies in the thin slab of points whose dot product with *v* has magnitude <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=O\left(\frac{1}{\sqrt{d}}\right)">”, which can be shown by fixing **v** as the first coordinate vector. From this it can be shown that, with high probability, two random points in the unit ball are nearly orthogonal. Specifically their vectors will be nearly orthogonal, they will be close to the surface, and have length <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=1 - O\left(\frac{1}{d}\right)">. Fixing the first as “north”, the second will have a projection of <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\pm O\left(\frac{1}{\sqrt{d}}\right)">, which is the same as their dot product. This means that with high probability the angle between the vectors will be <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{\pi}{2} \pm O\left(\frac{1}{\sqrt{d}}\right)">.

**In summary:**

1. As *d* ⟶ ∞ , the volume of the ball goes to zero
2. A unit ball’s volume is concentrated around its ‘equator’, i.e.: <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mid{x_1}\mid = O\left(\frac{1}{\sqrt{d}}\right)">
3. The vectors of two random points in the unit ball tend to be (nearly) orthogonal
4. The vectors tend to be close to the surface, with length <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=1 - O\left(\frac{1}{d}\right)">
5. The angles between the vectors tend to be <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{\pi}{2} \pm O\left(\frac{1}{\sqrt{d}}\right)">

Here is an attempt to explain it intuitively:

[https://mathoverflow.net/questions/210291/how-to-explain-the-concentration-of-measure-phenomenon-intuitively](https://mathoverflow.net/questions/210291/how-to-explain-the-concentration-of-measure-phenomenon-intuitively)

But a better way to visualise it actually comes from this explanation:

[https://www.quora.com/Why-is-the-higher-the-dimension-the-less-the-hypervolume-of-a-hypersphere-inscribed-in-a-hypercube-occupy-the-hypervolume-of-the-hypercube](https://www.quora.com/Why-is-the-higher-the-dimension-the-less-the-hypervolume-of-a-hypersphere-inscribed-in-a-hypercube-occupy-the-hypervolume-of-the-hypercube)

The distribution trend can be seen in these Python generated graphs:

![](Images/image_4.png)

![](Images/image_15.png)

![](Images/image_6.png)

![](Images/image_17.png)

[https://www.johndcook.com/blog/2017/07/13/concentration_of_measure/](https://www.johndcook.com/blog/2017/07/13/concentration_of_measure/)
 
 
The theorem for volume near the equator of the unit ball goes:

For *c* ≥ 1 and *d* ≥ 3, at least a <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=1 - \frac{2}{c} e^-\frac{c^2}{2}">   fraction of the volume of the d-dimensional unit ball has <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mid{x_1}\mid \le \frac{c}{\sqrt{d-1}}">  .

### Question: What does it mean that two coordinates are nearly orthogonal?

The theorem for near orthogonality goes:

Consider drawing n points *x*₁, *x*₂, … , *x*<sub>*n*</sub> at random from the unit ball. With probability <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=1 - O\left(\frac{1}{n}\right)">  

1. <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mid{x_i}\mid \ge 1 - \frac{2\,\mathrm{ln}\,n}{d}\,\,\text{for all}\,i \text{, and}"></div>  
2. <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mid{x_i}\cdot{x_j}\mid \le \frac{\sqrt{6\,\mathrm{ln}\,n}}{\sqrt{d-1}}\,\,\text{ for all}\,i \ne j">  
 
 
One way of measuring orthogonality is to calculate the squared dot product of a vector and other coordinate vectors throughout the unit ball. So for example if a random vector = (1, 0, 0, … 0), all these dot products give zero mean, with variance = <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{1}{d}">, and standard deviation = <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sqrt{\frac{1}{d}}">. To put it differently, the expected value of any two coordinate vectors' dot products is 0, because each component of the sum is the product of two independent Gaussians with mean 0. So we can conclude that in higher dimensions the inner dot product of any two coordinate vectors is likely to be 0, with high probability.
 
 
[https://math.stackexchange.com/questions/995623/why-are-randomly-drawn-vectors-nearly-perpendicular-in-high-dimensions](https://math.stackexchange.com/questions/995623/why-are-randomly-drawn-vectors-nearly-perpendicular-in-high-dimensions)

[https://math.stackexchange.com/questions/3059747/probability-of-two-random-points-being-orthogonal-in-higher-dimensional-unit-sph](https://math.stackexchange.com/questions/3059747/probability-of-two-random-points-being-orthogonal-in-higher-dimensional-unit-sph)

### Question: How do you generate points uniformly at random from a ball?
 
 
Generating points *x*₁, *x*₂, …, *x*<sub>*d*</sub> each with coordinates an independent Gaussian variable:
   using zero mean, unit variance Gaussian, i.e.
   <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{1}{\sqrt{2\pi}} e^-\frac{x^2}{2}">   on the real line

Gives probability density of *x*:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}} e^-\frac{x^{2}_{1} %2B x^{2}_{1} %2B \ldots %2B x^{2}_{d}}{2}">  

with spherical symmetry.

Normalising vector **x** = ( *x*₁, *x*₂, …, *x*<sub>*d*</sub> ) to a unit vector <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{\mathbf{x}}{\mid{\mathbf{x}}\mid}">

Gives a distribution that is uniform over the sphere’s surface
*Note:* once normalised, the coordinates are no longer independent

Generating a point **y** uniformly over the unit ball volume,
we must scale the surface point <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{\mathbf{x}}{\mid{\mathbf{x}}\mid}"> by a scalar *ρ* ∈ [0, 1] 

Gives a point:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=y = p\frac{\mathbf{x}}{\mid\mathbf{x}\mid}">  

*Note:* The distribution of *ρ* as a function of *r* is not uniform across the ball. Instead the density of *ρ* at distance *r* is proportional to *r*<sup>*d* - 1</sup> in *d* dimensions. Solving <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\int_{r=0}^{r=1} cr^{d-1}dr = 1">   we see that we should set *c* = *d*. Another way of looking at it is knowing that the density at radius *r* is exactly <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{d}{dr}(r^d V_d) = dr^{d-1}V_d"> . So we would want to pick *ρ*(*r*) with density equal to *dr*<sup>*d*-1</sup> for *r* over [0,1].

[https://stats.stackexchange.com/questions/85916/distribution-of-scalar-products-of-two-random-unit-vectors-in-d-dimensions](https://stats.stackexchange.com/questions/85916/distribution-of-scalar-products-of-two-random-unit-vectors-in-d-dimensions)

On a more practical note, there are a number of ways to sample from a unit ball. Rejection sampling is a common and intuitive approach, but it is not very efficient. Two of the most well known, much more efficient ways, are the Box-Muller transform and the inverse transform sampling.

The Box-Muller transform takes two uniformly distributed random numbers and derives Gaussian distributed random numbers.

[https://www.youtube.com/watch?v=EXsdT91XFAY](https://www.youtube.com/watch?v=EXsdT91XFAY)

Inverse transform sampling requires that the CDF is known, and entities have to be normalised. Most importantly it doesn’t generalise well to higher dimensional problems, due to difficulties in calculating a CDF. Unlike rejection sampling, however, it is 100% efficient.

[https://www.youtube.com/watch?v=rnBbYsysPaU](https://www.youtube.com/watch?v=rnBbYsysPaU)

This blog post discusses a variety of ways with some pseudo code that is fairly easy to implement.

[http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/](http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/)

### Question: How do Gaussians behave in Higher Dimensions?
 
 
The *d*-dimensional spherical Gaussian with zero mean and variance *σ*² has the following density function: 
 
<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}\sigma^d}\mathrm{exp}(-\frac{\mid\mathbf{x}\mid^2}{2\sigma^2})">  

Although density is maximum at origin, there is little volume. The radius needs to be increased to around √d before there is significant and hence probability mass. Beyond √d the probability density starts to drop off at a much faster rate than the volume increases.

### What is a Random Projection?

Imagine we need to solve the following applied problem:

We are given a database with n points in <i>R<sup>d</sup></i>, where both *n* and *d* are large. The database is likely to be preprocessed and stored in an efficient data structure. The purpose of the database is to answer queries where a query point is provided, and the nearest or approximately nearest point to it has to be found. The number of queries will be large, so the time to answer a query should be small, ideally a simple function of log *n* and log *d*. Preprocessing time could take longer, for example a polynomial function of *n* and *d*.

For this type of problem dimensionality reduction, where we project the database points in *d*-dimensions to a *k*-dimensional space where *k* ≪ *d*, can be very helpful as long as relative distances between points are approximately preserved.

For the projection *f* : <b>R</b><sup><i>d</i></sup> ⟶ <b>R</b><sup><i>k</i></sup> pick *k* Gaussian vectors <b>u<sub>1</sub>, u<sub>2</sub>, ..., u<sub>k</sub></b> in <b>R</b><sup><i>d</i></sup>. For any vector **v**, define the projection *f*(**v**) as:

<p style="text-align: center;"><i>f</i>(<b>v</b>) = ( <b>u<sub>1</sub></b> · <b>v</b>, <b>u<sub>2</sub></b> · <b>v</b>, ..., <b>u<sub>k</sub></b> · <b>v</b> )</p>

It can be shown that, with high probability, <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mid{f}(\mathbf{v})\mid \approx \sqrt{k}\mid\mathbf{v}\mid">. For any two vectors <b>v<sub>1</sub></b> and <b>v<sub>2</sub></b>, *f*( <b>v<sub>1</sub></b> - <b>v<sub>2</sub></b> ) = *f*( <b>v<sub>1</sub></b> ) - *f*( <b>v<sub>2</sub></b> ).

So to estimate the distance |<b>v<sub>1</sub></b> - <b>v<sub>2</sub></b>| in <b>R</b><sup><i>d</i></sup> it suffices to calculate *f*( <b>v<sub>1</sub></b> ) - *f*( <b>v<sub>2</sub></b> ) <i>in *k*-dimensional space</i> and divide by <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sqrt{k}"> (which is known).

Note that the distances increase when projected into a lower dimensional space, because the vectors <b>u<sub>i</sub></b> are not unit length. Neither are they orthogonal, which would have forfeited statistical independence.

**Random Projection Theorem:**

<i>Let </i><b>v</b><i> be a fixed vector in </i><b>R</b><sup><i>d</i></sup><i> and f</i>(<b>v</b>) = ( <b>u<sub>1</sub></b> · <b>v</b>, <b>u<sub>2</sub></b> · <b>v</b>, ..., <b>u<sub>k</sub></b> · <b>v</b> ). <i>There exists a constant c ></i> 0 <i>such that for ε ∈</i> (0, 1), 

<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=P\big(\big|\mid{f}(\mathbf{v})\mid - \sqrt{k}\mid\mathbf{v}\mid\big| \ge \epsilon\sqrt{k}\mid\mathbf{v}\mid\big) \le 3e^{-ck\epsilon^{2}}"></p>

<i>where the probability is taken over the random draws of vectors <b>u<sub>i</sub></b> used to construct f.</i>

The theorem can be proven by first scaling the inner equality to show |**v**| = 1, then deriving *Var*(<b>u<sub>i</sub></b> · <b>v</b>) = 1 from  the fact the random variable <b>u<sub>i</sub></b> · <b>v</b> has Gaussian density. Then, applying the Gaussian Annulus Theorem with *k* instead of *d* using the independent Gaussian random variables <b>u<sub>1</sub></b> · <b>v</b>, <b>u<sub>2</sub></b> · <b>v</b>, ..., <b>u<sub>i</sub></b> · <b>k</b>, the theorem follows.

### What is the Johnson-Lindenstrauss Lemma?

It is typical for the running time of an algorithm to depend exponentially on the number of dimensions, *d*. The Johnson-Lindenstrauss lemma reduces the dimension *d* to log(*d*), and the algorithm can then run in time polynomial in *d*.

The Johnson-Lindenstrauss lemma also gives guarantees about the pointwise distance between all the points in the projection from *d* to log(*d*) (useful in for example k-nearest neighbour algorithm), preserving them approximately.

"Proposition (the JL lemma): For some <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=k \ge \frac{\mathrm{log}\,m}{\epsilon^{\small 2}}"> (where <span style="font-size:1.2em;">𝜀</span> is our chosen error tolerance, and <span style="font-size:1.2em;">𝑚</span> is the number of points), with high probability, map <span style="font-size:1.2em;">𝑓 : ℝ<sup>𝑑</sup> → ℝ<sup>𝑘</sup></span> does not change the pairwise distance between any two points more than a factor of (1 ± <span style="font-size:1.2em;">𝜀</span>), after scaling by <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sqrt\frac{n}{k}">."

https://www.quora.com/What-is-a-simplified-explanation-and-proof-of-the-Johnson-Lindenstrauss-lemma

**Johnson-Lindestrauss Lemma:**

<i>For any </i>0<i> < ε < </i>1<i> and any integer n, let <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=k \ge \frac{3}{c\epsilon^2}\mathrm{ln}\,n">  with *c* a constant defined as part of the probability mass upper bound in the Gaussian Annulus Theorem: <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=3e^{-c\beta^2}"></i>. <i>For any set of n points in R<sup>d</sup>, the random projection f : R<sup>d</sup> ⟶ R<sup>k</sup> defined above has the property that for all pairs of points </i>v<sub>i</sub><i> and </i>v<sub>j</sub>, with probability at least <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=1 - \frac{3}{2n}"></i>

**Johnson-Lindenstrauss Theorem, version by Achlioptas:**  

<i>Given x<sub>1</sub>, ..., x<sub>n</sub> ∈ R<sup>d</sup>, let X be the d x n matrix with the vectors x<sub>i</sub> as columns  
Let ε, β > 0 and  <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=k = \mathrm{log}(n) \cdot \frac{4 %2B 2\beta}{\frac{\epsilon^{\small  2}}{2} - \frac{\epsilon^{\small 3}}{3}}">  
Let R be a k x d matrix with independent entries ±1 as described above  
Construct the projected data points <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=Y := \frac{{R}\cdot{X}\cdot{1}}{\sqrt{k}}">  
Then, with probability at least 1 - n<sup>-β</sup>, property (*), i.e. distance, from the Johnson-Lindenstrauss theorem holds for all pairs of points (x<sup>i</sup>, x<sup>j</sup>)  </i>

https://www.youtube.com/watch?v=j9qbuGSjzeE

The gist of the proof:

"the JL lemma follows from the fact that the squared length of a vector is sharply concentrated around its mean when projected onto a random 𝑘-dimensional subspace"

https://www.quora.com/What-is-a-simplified-explanation-and-proof-of-the-Johnson-Lindenstrauss-lemma

A different way of stating it:

"the johnson-lindenstrauss lemma is true because the square of the dot product between a random vector and any vector is, in expectation, equal to the norm of that vector. You can then improve the estimate by using more random vectors until the variance is smaller than any fixed threshold, with high probability."

https://www.quora.com/What-is-a-simplified-explanation-and-proof-of-the-Johnson-Lindenstrauss-lemma

More formally:

By applying the Random Projection Theorem we can say that for any fixed <b>v<sub>i</sub></b> and <b>v<sub>j</sub></b> the probability that |*f*(<b>v<sub>i</sub></b> - <b>v<sub>j</sub></b>)| is outside the range

<p align="center"><img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\big[(1 - \epsilon)\sqrt{k}\mid\mathbf{v_i} - \mathbf{v_j}\mid\,,(1 %2B \epsilon)\sqrt{k}\mid\mathbf{v_i} - \mathbf{v_j}\mid\big]"></p>

is at most <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=3e^{-ck\epsilon^2} \le \frac{3}{n^{\small 3}}\,\,\mathrm{for}\,k \ge \frac{3\,\mathrm{ln}\,n}{c\epsilon^2}">. Since there are <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=(^n_2) < \frac{n^2}{2}"> pairs of points, by the union bound, the probability that any pair has a large distortion is less than <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{3}{2n}">. (Blum et al)

Note that this is the stronger conclusion of the theorem and ensures that it holds for all <b>v<sub>i</sub></b> and <b>v<sub>j</sub></b> (not just most of them). This can be important in an algorithm like nearest neighbours. For this dimension reduction the dominant term is typically 1/*ε*<sup>2</sup>.

### Question: What is the connection between Random Projection and the Johnson-Lindenstrauss lemma?

The random projection theorem says that projecting from dimensions *d* to *k*, the probability that the length of the projection of a single vector differs significantly from its expected value is exponentially small.  

More specifically, by a union bound, the probability that any of *O*(*n*<sup>2</sup>) pairwise differences |<b>v<sub>i</sub></b> - <b>v<sub>j</sub></b>| among *n* vectors <b>v<sub>1</sub></b>, ... ,<b>v<sub>2</sub></b> differs slightly from their expected values is small, provided <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=k \ge \frac{3}{c\epsilon^2}">.  

Thus the random projection, with high probability preserves all relative pairwise distances between points in a set of *n* points. This leads directly to the Johnson-Lindenstrauss lemma. (Blum et al)

### Question: How do you identify which Gaussian a point belongs to when there is more than one distribution?
 
 
The algorithm is ultimately simple:
- Calculate the distance between pairs of points
- Points whose distance apart is smaller are from the same Gaussian (vs points further apart)

Firstly, from the Gaussian Annulus Theorem, which states that for large *d*, the *d*-dimensional Gaussian is located in the annulus with high probability, we can derive that for two points **x** and **y**:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mid\mathbf{x} - \mathbf{y}\mid = \sqrt{2d} \pm O(1)">

If we now have two Gaussians, with centres **p** and **q**, separated by distance ∆, then if point **x** is drawn from the first and point **y** from the second Gaussian, then the distance between them will be close to <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sqrt{\triangle^2 %2B 2d}"> since **x** - **p**, **p** - **q**, **q** - **y** are mutually (nearly) perpendicular. It can then be show that:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mid\mathbf{x} - \mathbf{y}\mid^2 = \triangle^2 %2B 2d \pm O(\sqrt{d})">  

In order to ensure that two points picked from the same Gaussian are closer to each other than two points picked from different Gaussians requires that the upper limit of the distance between a pair of points from the same Gaussian is at most the lower limit of distance between point from different Gaussians.

It can be derived from the above that spherical Gaussians can be separated this way as long as their centres are separated by *ω*( *d*¹ᐟ⁴).

Refined algorithm:
- Calculate all pairwise distances between points
- The cluster of smallest pairwise distance must be from a single Gaussian
- Remove these points
- The remaining points come from the second Gaussian

### Question: How do you fit a spherical Gaussian to data?
 
 
A quick review of PDF and CDF is in order. Remember, the PDF is the gradient of the CDF, and inversely the CDF is the integral of the PDF, a.k.a. the Area Under the Curve (AUC). The total area under the PDF equals 1 (between -∞ and ∞).

In a Gaussian distribution, the parameters we are trying to discover in a sample of data can be determined as follows:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\hat{\mu} = \frac{1}{N}\displaystyle\sum_{n=1}^{N}\mathbf{x}_n"></div>  

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sigma^2 = \frac{1}{N}\displaystyle\sum_{n=1}^{N}(\mathbf{x}_n - \hat{\mu})^2">  
 

A distribution’s width scales with the standard deviation (not the variance). So if the two variances *σ*² = 4 and *σ*² = 1, then the width in the first distribution will be twice that in the second.

In *d* dimensions:

**µ** = ( *µ*₁, *µ*₂, … *µ*<sub>d</sub> )<sup>*T*</sup>  
**Σ** = ( *σ*ᵢⱼ ) 
i.e. **Σ** is a covariance matrix ( *d*-by-*d* square matrix) with element *σ*ᵢⱼ at row *i* and column *j*


<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=p(\mathbf{x}\mid\mathbf{\mu},\mathbf{\Sigma}) = \frac{1}{(2\pi)^{d/2}\mid\mathbf{\Sigma}\mid^{1/2}}\mathrm{exp}\big(-\frac{1}{2}(\mathbf{x} - \mathbf{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x} - \mathbf{\mu})\big)">  

Values along the leading diagonal gives the variance of each variable, and off-diagonal values measure correlations between variables.
 
 
The mean vector **µ** is the expectation of **x**:  
**µ** = *E*( **x** )

The covariance matrix **Σ** is the expectation of the deviation of **x** from the mean:

<i>E</i>[ ( **x** - **µ** ) ( **x** - **µ** )*ᵀ* ]

**Σ** = ( *σᵢⱼ* ) = **Σ**<i>ᵀ</i> is a symmetric *d x d* matrix 

Any covariance matrix is *positive semi-definite*. The same goes for its inverse, if it exists.
This means: 

**x**<i>ᵀ</i> **Σ**<i>x</i> ≥ 0 for any real valued vector **x**
 
 
*σ*ᵢⱼ is not the standard deviation, but the covariance between *i* and *j* in **x**. Eg. in the 1-dimensional case, *σ*₁₁ = *σ*².
The sign of *σᵢⱼ* (the covariance) helps to determine the relationship between two variables:
- If *xⱼ* is large when *xᵢ* is large, then ( *xⱼ* − *μⱼ* )( *xᵢ* − *μᵢ* ) will tend to be positive;
- If *xⱼ* is small when *xᵢ* is large, then ( *xⱼ* − *μⱼ* )( *xᵢ* − *μᵢ* ) will tend to be negative.
Large covariance probably implies redundant information.
On the other hand *σᵢⱼ* = 0 implies statistical independence.

The correlation coefficient, a.k.a. Pearson correlation coefficient, is a value between -1 and 1 obtained by normalising the covariance *σᵢⱼ* by the square root of the product of the variances *σᵢᵢ* and *σⱼⱼ*
 
 <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\rho(x_i, x_j) = \rho_{ij} = \frac{{\Large \sigma}_{\small ij}}{\sqrt{ \Large \sigma_{\small ii}\sigma_{\small jj}}}">  


Consider 2-dimensional Gaussian with:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mathbf{\mu} = \left( {\begin{array}{cc} 0 \\ 0 \end{array} } \right)">  

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mathbf{\Sigma} = \left( {\begin{array}{cc} 1\hspace{4mm}0 \\ 0\hspace{4mm}1 \end{array} } \right)">  
 

The means are zero and the variances are equal. This will give a /spherical Gaussian/ that can be plotted as follows:

![](Images/image_5.png)

Consider another 2-dimensional Gaussian with:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mathbf{\mu} = \left( {\begin{array}{cc} 0 \\ 0 \end{array} } \right)">  

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mathbf{\Sigma} = \left( {\begin{array}{cc} 1\hspace{4mm}0 \\ 0\hspace{4mm}4 \end{array} } \right)">  


The means are zero, but the variances are not the same. This will give an elliptical Gaussian that can be plotted as follows:

![](Images/image_16.png)

Consider another 2-dimensional Gaussian with:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mathbf{\mu} = \left( {\begin{array}{cc} 0 \\ 0 \end{array} } \right)">  

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mathbf{\Sigma} = \left( {\begin{array}{cc} 1\hspace{4mm}-1 \\ -1\hspace{4mm}4 \end{array} } \right)">  


The means are zero, but the variances are not the same, and the off-diagonals are non-zero. This will give an elliptical Gaussian that can be plotted as follows:

![](Images/image_8.png)

It is useful to step through the links between the probability density function, cumulative distribution function, and a spherical Guassian provided in the below notes:

[http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note08-2up.pdf](http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note08-2up.pdf)

More formally, as per Blum et al, given the sample points **x₁**, **x₂**, …, **x**<sub>*n*</sub> in *d* dimensions we want to find the spherical Gaussian that best fits these points. If *f* is the unknown Gaussian (with mean *µ* and variance *σ*² ) then the probability density for picking these points when sampling according to *f* is given by:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=c\mathrm{exp}\left(-\frac{(\mathbf{x_1} - \mathbf{\mu})^2 %2B (\mathbf{x_2} - \mathbf{\mu})^2 %2B \ldots %2B (\mathbf{x_n} - \mathbf{\mu})^2}{2\sigma^2}\right)">  

*c* is a normalising constant, the reciprocal of <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\left[\int e^-\frac{\mid\mathbf{x} - \mathbf{\mu}\mid^2}{2\sigma^2}dx\right]^n">  
which is equivalent to <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\left[\int e^-\frac{\mid\mathbf{x}\mid^2}{2\sigma^2}dx\right]^{-n}">  as origin can be *µ* when integrating over -∞ to ∞
and that is equal to: <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\frac{1}{\large (2\pi)^{\frac{\small n}{\small 2}}}">  

Now we are looking to find the *Maximum Likelihood Estimator* (MLE) of our unknown Gaussian *f*, in other words the *f* that maximises the above probability density.

The lemma states that ( **x₁** - **µ** )² + ( **x₂** - **µ** )² + … + ( **x**<sub>*n*</sub> - **µ** )² is minimised when **µ** is the centroid of the points **x₁**, **x₂**, …, **x**<sub>*n*</sub> , in other words:

<img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\mathbf{\mu} = \frac{1}{n}(\mathbf{x_1} %2B \mathbf{x_1} %2B \ldots %2B \mathbf{x}_n)">  

To then calculate the MLE of *σ*² for *f*, set **µ** as true centroid and *σ* as standard deviation of the sample, and substitute <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=v = \frac{1}{2\sigma^2}"> and *a* = ( **x₁** - **µ** )² + ( **x₂** - **µ** )² + … + ( **x**<sub>*n*</sub> - **µ** )² into the formula for picking probability density. We can then go on to show that the maximum occurs when <img style="float: right;" src="https://render.githubusercontent.com/render/math?math=\sigma = \frac{\sqrt{a}}{\sqrt{nd}}">  .

This produces the following lemma:

“The maximum likelihood spherical Gaussian for a set of samples is the Gaussian with centre equal to the sample mean and standard deviation equal to the standard deviation of the sample from the true mean.” - Blum et al

### Question: What is the difference between the Standard error and Standard deviation?

Standard deviation quantifies the variation within a set of measurements.
Standard error quantifies the variation of the means of multiple sets of measurements.

[https://www.youtube.com/watch?v=A82brFpdr9g](https://www.youtube.com/watch?v=A82brFpdr9g)
