<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/><meta name="exporter-version" content="Evernote Mac 6.1.1 (452253)"/><meta name="altitude" content="16.15255737304688"/><meta name="author" content="maartenslo@hotmail.com"/><meta name="created" content="2020-06-09 09:42:17 +0000"/><meta name="latitude" content="51.53826929153488"/><meta name="longitude" content="-0.07351958231742141"/><meta name="source" content="desktop.mac"/><meta name="updated" content="2020-07-01 14:16:21 +0000"/><title>Foundations of Data Science</title></head><body>
<div><b><span style="font-size: 18px;">Chapter 2 : High-Dimensional Space</span></b></div>
<div><span style="font-size: 18px;"><br/></span></div>
<div><i><span style="font-size: 18px;">Question: What is a "Zero mean, unit variance Gaussian”?</span></i></div>
<div><span style="font-size: 18px;"><br/></span></div>
<div><b><span style="font-size: 16px;">Gaussian distribution</span></b></div>
<div><span style="font-size: 12px;"><br/></span></div>
<div><a href="https://www.inf.ed.ac.uk/teaching/courses/mlpr/2017/notes/w2b_univariate_gaussian.pdf">https://www.inf.ed.ac.uk/teaching/courses/mlpr/2017/notes/w2b_univariate_gaussian.pdf</a></div>
<div><br/></div>
<div>Math notation: X ~ N( <i>μ</i>, <i>σ</i> )</div>
<div>                         X ~ N( 0, 1 )</div>
<div>CS notation: x = randn( )</div>
<div><br/></div>
<div>Function: </div>
<div style="text-align: center"><img src="Foundations%20of%20Data%20Science.resources/Probability%20Density%20Function%201.png" height="114" width="308"/></div>
<div style="text-align: center"><br/></div>
<div>a.k.a. Probability Density Function (PDF) for Standard Normal Distribution</div>
<div><br/></div>
<div>The <i>inflection point</i> of the Gaussian (normal) curve is one standard deviation below the mean, and one standard deviation above the mean.</div>
<div><br/></div>
<div>Let there be a curve with mean <i>μ</i> and standard deviation <i>σ</i></div>
<div>Then inflection points occur where <i>x = μ ± σ</i></div>
<div><i><br/></i></div>
<div><i>μ</i> : Mu</div>
<div><i>σ</i> : lower case sigma</div>
<div><br/></div>
<div><a href="https://www.thoughtco.com/inflection-points-of-a-normal-distribution-3126446">https://www.thoughtco.com/inflection-points-of-a-normal-distribution-3126446</a></div>
<div><br/></div>
<div>One property of the inflection point is that the second derivative of the function will be zero (however, not all points where the 2nd derivative of the function is zero, will be an inflection point). </div>
<div><br/></div>
<div>So for a graph <i>y = f( x )</i></div>
<div>Given an inflection point at <i>x = a</i></div>
<div>Then <i>f’’( a )</i> = 0</div>
<div><br/></div>
<div>This can be proven using the Probability Density Function, as illustrated in the reference document above. The PDF is generally used in the context of continuous (rather than discrete) random variables. It can be used to specify the probability that a random variable falls <i>within a particular range of values</i>, as opposed to taking on a particular value.</div>
<div><br/></div>
<div>See <a href="https://en.wikipedia.org/wiki/Probability_density_function">https://en.wikipedia.org/wiki/Probability_density_function</a></div>
<div><br/></div>
<div><i>Variance</i> (for continuous variables) can be expressed as:</div>
<div><br/></div>
<div><i>σ</i><sup>2</sup> = Σ ( <i>X<sub>i</sub></i> - <i>X</i> )<sup>2</sup> / <i>N</i></div>
<div>Where <i>σ</i><sup>2</sup> = variance</div>
<div>( <i>X<sub>i</sub></i> - X )<sup>2</sup> = (Individual Value – Mean)<sup>2</sup></div>
<div>Σ = Summation of function associated with it</div>
<div><i>N</i> = Total number of data points in our dataset</div>
<div><br/></div>
<div>Can also be expressed as: <i>σ</i><sup>2</sup> = Σ ( <i>x<sub>i</sub></i> - <i>μ</i> )<sup>2</sup> / <i>N</i></div>
<div><br/></div>
<div>For discrete variables it is different, but the above is relevant in the context of the Gaussian curve.</div>
<div><br/></div>
<div><a href="http://zerosnones.net/variance-limitation-properties-applications/">http://zerosnones.net/variance-limitation-properties-applications/</a></div>
<div><br/></div>
<div><b>"Unit variance</b> means that the standard deviation of a sample as well as the <b>variance</b> will tend towards 1 as the sample size tends towards infinity." This is typically referred to as the standard normal distribution.</div>
<div><br/></div>
<div>For a sample of a million random values:</div>
<div><br/></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">import numpy as np</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">from matplotlib import pyplot as plt</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">N = int(1e6) # 1e6 is a float, numpy wants int arguments</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">xx = np.random.randn(N)</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">hist_stuff = plt.hist(xx, bins=100)</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">plt.show()</span></div>
<div><br/></div>
<div><img src="Foundations%20of%20Data%20Science.resources/EA204A93-19C5-4D07-BB61-8949295DAA30.png" height="252" width="387"/></div>
<div>
<div style="border-right-width: 30px; min-height: 45px; padding-right: 0px; padding-bottom: 0px;">
<div>
<div style=" outline: currentcolor none medium;">
<div><br/></div>
</div>
</div>
</div>
</div>
<div><br/></div>
<div style="font-weight: bold;"><span style="font-size: 16px;">Variance and Standard Deviation</span></div>
<div><br/></div>
<div><a href="https://www.quora.com/What-are-zero-mean-unit-variance-Gaussian-random-numbers">https://www.quora.com/What-are-zero-mean-unit-variance-Gaussian-random-numbers</a></div>
<div><br/></div>
<div><i>Variance</i> and <i>standard deviation</i> have a close relationship. The <i>standard deviation</i> is simply the square root of the <i>variance</i>.</div>
<div><br/></div>
<div>We can see this quite easily:</div>
<div><br/></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">import numpy as np</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;"><br/></span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">a = [11,9,5,13,18,6,9,12,10,7]</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;"><br/></span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">mean = np.sum(a) / len(a)</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">sqr_dev = []</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">for i in a:</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">    sqr_dev.append(np.square(abs(mean - i)))</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;"><br/></span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">variance = np.sum(sqr_dev) / len(a)</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">std_dev = np.sqrt(variance)</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;"><br/></span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">print(f"mean: {mean}")</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">print(f"original: {a}")</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">print(f"squared deviations: {sqr_dev}")</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">print(f"variance: {variance}")</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">print(f"standard deviation: {std_dev}")</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">print(f"np variance: {np.var(a)}")</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">print(f"np standard deviation: {np.std(a)}")</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;"><br/></span></div>
<div><span style="font-family: Arial;">To conclude, a “zero mean, unit variance Gaussian” is a Gaussian distribution with <i>µ</i></span><span style=""><span style=""><span style=" clip: rect(1.696em, 1002.35em, 2.939em, -1000em); top: -2.564em; left: 0em;"><span style="font-family: Arial;"><span style="padding-left: 0.313em;">=</span></span><span style="padding-left: 0.313em;"><span style="font-family: Arial;">0,</span></span></span></span></span> <i>σ =</i> 1<span style="font-family: &quot;Arial Black&quot;;"><span style="font-family: Arial;"><span style=""><span style=""><span style=" clip: rect(1.696em, 1002.27em, 2.767em, -1000em); top: -2.564em; left: 0em;"><span style="padding-left: 0.313em;"><span style="font-size: 17px;">,</span> also called a Standard Normal Distribution.</span></span></span></span></span></span></div>
<div><span style="font-family: Arial;"><br/></span></div>
<div><span style="font-family: Arial;">Note that the Standard Normal Distribution is simply one kind of Normal Distribution. More generally, "</span>the <b>normal distribution</b> is a probability function that describes how the values of a variable are <b>distributed</b>” (Wikipedia). The mean and variance <font face="Arial">could have different values, for example it could have mean 4 and variance 2, as in the following example:</font></div>
<div><span style="font-family: Arial;"><br/></span></div>
<div>
<div style="border-right-width: 30px; min-height: 79px; padding-right: 0px; padding-bottom: 0px;">
<div>
<div style=" outline: currentcolor none medium;">
<div><span style="font-family: &quot;Andale Mono&quot;;">import numpy as np</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">from matplotlib import pyplot as plt​</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;"><br/></span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">x = 4 + (np.random.randn(1000000) * np.sqrt(2))</span></div>
</div>
</div>
</div>
<div style=" border-bottom: 0px solid transparent; top: 79px;"/>
</div>
<div><span style="font-family: &quot;Andale Mono&quot;;">print(x.mean()</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">print(x.var())</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">print(x.std())</span></div>
<div><br/></div>
<div>4.000612096576726</div>
<div>2.0002004340052753</div>
<div>1.4142844247198918</div>
<div><br/></div>
<div>
<div style="border-right-width: 30px; min-height: 45px; padding-right: 0px; padding-bottom: 0px;">
<div>
<div style=" outline: currentcolor none medium;">
<div><span style="font-family: &quot;Andale Mono&quot;;">h = plt.hist(x, bins=100)</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">plt.show()</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;"><br/></span></div>
</div>
</div>
</div>
<div style=" border-bottom: 0px solid transparent; top: 45px;"/>
</div>
<div><img src="Foundations%20of%20Data%20Science.resources/FAE8345E-F935-4177-AC03-406430CA4440.png" height="248" width="387"/></div>
<div><br/></div>
<div>So to return to our initial function:</div>
<div style="text-align: center"><img src="Foundations%20of%20Data%20Science.resources/Probability%20Density%20Function%201.png" height="114" width="308"/></div>
<div><br/></div>
<div>"The factor <span style="">1 / √2<i>π</i> </span> in this expression ensures that the total area under the curve <i>f</i>(<i>x</i>) is equal to one.<sup><span style="font-size: 12px;"><a href="https://en.wikipedia.org/wiki/Normal_distribution#cite_note-4"> </a></span></sup>The factor <span style="">1 / 2</span> in the exponent ensures that the distribution has unit variance (i.e., the variance is equal to one), and therefore also unit standard deviation. This function is symmetric around <span style=""><i>x</i> = 0</span>, where it attains its maximum value <span style="">1 / √2<i>π </i></span>and has <a title="Inflection point">inflection points</a> at <span style=""><i>x</i> = + 1</span> and <span style=""><i>x</i> = − 1.” - Wikipedia</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;"><br/></span></div>
<div><i><span style="font-size: 18px;">Question: How do you shift a distribution to zero mean?</span></i></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div><b><i>Answer:</i></b> Subtract the mean from (each value in) the distribution, and divide by the variance.</div>
<div><br/></div>
<div>
<div style="border-right-width: 30px; min-height: 96px; padding-right: 0px; padding-bottom: 0px;">
<div>
<div style=" outline: currentcolor none medium;">
<div><span style="font-family: &quot;Andale Mono&quot;;">import numpy as np</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">from matplotlib import pyplot as plt</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">​</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">x = 10 + (np.random.randn(1000000))</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">normalised_x = (x - x.mean()) / x.std()</span></div>
</div>
</div>
</div>
<div style=" border-bottom: 0px solid transparent; top: 96px;"/>
</div>
<div>
<div style="border-right-width: 30px; min-height: 96px; padding-right: 0px; padding-bottom: 0px;">
<div>
<div style=" outline: currentcolor none medium;">
<div><span style="font-family: &quot;Andale Mono&quot;;">h = plt.hist(x, bins=100)</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">plt.show()</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">​h = plt.hist(normalised_x, bins=100)</span></div>
<div><span style="font-family: &quot;Andale Mono&quot;;">plt.show()</span></div>
</div>
</div>
</div>
<div style=" border-bottom: 0px solid transparent; top: 96px;"/>
</div>
<div>
<div>
<div><img src="Foundations%20of%20Data%20Science.resources/D8B5700B-4787-4EE6-A428-8F5E34DEFC03.png" height="248" width="387"/></div>
</div>
</div>
<div><br/></div>
<div><img src="Foundations%20of%20Data%20Science.resources/B126AA9A-7DE4-46B4-A488-869C731577A3.png" height="248" width="387"/></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>Given that in the above example the mean is ten, which we added to the dataset, and the variance should be close to 1 (unit variance), the formula is in effect simply reversing that operation: subtracting the mean, and dividing by the variance (≈1).</div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: What is the Empirical Rule?</i></span></div>
<div><br/></div>
<div>It is also known as the <i>68–95–99.7 rule</i><b>.</b> </div>
<div><br/></div>
<div>It implies that 68.27%, 95.45% and 99.73% of the values lie within one, two and three standard deviations of the mean.</div>
<div><br/></div>
<div>Mathematically it can be expressed as follows:</div>
<div><br/></div>
<div>Pr( <i>µ</i> - 1<i>σ</i> ≤ <i>X</i> ≤ <i>µ</i> + 1<i>σ </i>) ≈ 0.6827</div>
<div>Pr( <i>µ</i> - 2<i>σ</i> ≤ <i>X</i> ≤ <i>µ</i> + 2<i>σ </i>) ≈ 0.9545</div>
<div>Pr( <i>µ</i> - 3<i>σ</i> ≤ <i>X</i> ≤ <i>µ</i> + 3<i>σ </i>) ≈ 0.9973</div>
<div><br/></div>
<div>The <i>three-sigma rule</i> states that even in the case of non-normally distributed variables, 88.8% of cases will still fall within the three sigma intervals (three standard deviations from the mean). This follows from <i>Chebyshev’s inequality</i>, which will be looked at later.</div>
<div><br/></div>
<div>The <i>three-sigma rule of thumb</i> is the observation that, for many cases (eg. in social sciences). most cases fall within three standard deviations of the mean. </div>
<div><br/></div>
<div><i>Confidence intervals</i> can be calculated to provide a range of plausible values for the unknown parameter in question. The interval has an associated <i>confidence level. </i></div>
<div><i><br/></i></div>
<div><a href="https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule">https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule</a></div>
<div><br/></div>
<div>The Gaussian has an important property in high dimensions. If we were to generate <i>n</i> points in <i>d</i>-dimensions, and each coordinate is a zero mean, unit variance Gaussian, and <i>d</i> is sufficiently large, then the <i>distance</i> between every pair of points will be almost exactly the same, with high probability.</div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div><span style="font-size: 18px;"><i>Question: What is a “Unit ball”?</i></span></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>A <i>unit circle</i> is the set of points of distance one from a fixed central point in two dimensions, in other words a circle with radius one. A <i>unit sphere</i> is the set of points in three dimensions of distance 1 from a fixed central point, in other words a sphere with radius one. The unit sphere is also known as a <i>unit ball</i>.</div>
<div><br/></div>
<div>Any sphere can be transformed to a unit sphere through a combination of <i>translation</i> and <i>scaling</i> (Euclidian geometric transformations).</div>
<div><br/></div>
<div><a href="https://en.wikipedia.org/wiki/Unit_sphere">https://en.wikipedia.org/wiki/Unit_sphere</a></div>
<div><br/></div>
<div>The unit ball has important properties in higher dimensions.</div>
<div><br/></div>
<div>The <i>volume</i> of a high-dimensional unit ball is concentrated near its surface and also at its equator. One consequence of this is that the set of all points <i>x</i> such that |<i>x</i>| ≤ 1 goes to zero as <i>d</i> goes to infinity.</div>
<div><br/></div>
<div><i><span style="font-size: 18px;">Question: What is the Expected Value?</span></i></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>Intuitively the expected value is simply the value that can be expected following some kind of action or event. In statistics and probability, "the <em>expected value</em> is calculated by multiplying each of the possible outcomes by the likelihood each outcome will occur and then summing all of those <em>values</em>.” - <a href="https://www.investopedia.com/terms/e/expected-value.asp">Investopedia</a></div>
<div><br/></div>
<div>It also has a connection to the Central Limit Theorem in that the “expected value is a measure of <strong>central tendency</strong>; a value for which the results will tend to. When a probability distribution is normal, a plurality of the outcomes will be close to the expected value.” - <a href="https://brilliant.org/wiki/expected-value/">Brilliant</a></div>
<div><br/></div>
<div>The simple version of the formula is for binomial events, which simply states that you multiply the probability with the number of times that the event occurs:</div>
<div><br/></div>
<div><i>E( X ) = Pr( x ) * X</i></div>
<div><br/></div>
<div><a href="https://www.youtube.com/watch?v=lxYBCrrhLW0">https://www.youtube.com/watch?v=lxYBCrrhLW0</a></div>
<div><br/></div>
<div>However, this is often too simple, so we need a way to represent multiple types of events. The formula in this case looks as follows:</div>
<div><br/></div>
<div><i>E( X ) = ∑X * Pr( X )</i></div>
<div><br/></div>
<div><a href="https://www.youtube.com/watch?v=_eIZKor-h48">https://www.youtube.com/watch?v=_eIZKor-h48</a></div>
<div><br/></div>
<div>These formulas are for discrete random variables, in other words where there are a countable number of possible values (even it is countably infinite). Each of these values have a probability between 0 and 1, and the sum of all the probabilities is equal to 1.</div>
<div><br/></div>
<div>There is also the continuous case, and the general case. The differences between these types of random variables require a more rigorous exploration than will be done in this section.</div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: What is the Law of Large Numbers?</i></span></div>
<div><br/></div>
<div>Informally, assuming we have a fair way of sampling, it simply means that the probability of reaching our expected value approaches 1 as the number of samples increases to infinity. A different way of putting is that the mean of the sample will approach the true mean of the population as <i>n </i>→ ∞.</div>
<div><br/></div>
<div>A simple example is the tossing of a coin. As the number of coin tosses <i>n</i> increases, the likelihood of us having tossed 50% heads and 50% tails tends towards 1. </div>
<div><br/></div>
<div>What it <i>does not mean</i> is that if we have a streak of heads that the probability of throwing tails suddenly increases. This is called the <i>Gambler’s Fallacy.</i> That is because<i> </i>the probability has no historical dependency, in other words it is <i>statistically independent</i>.</div>
<div><i><br/></i></div>
<div><a href="https://en.wikipedia.org/wiki/Gambler's_fallacy">https://en.wikipedia.org/wiki/Gambler's_fallacy</a></div>
<div><br/></div>
<div>There is an important distinction between the Weak and the Strong Law of Large Numbers. The Weak Law essentially says that there is no guarantee that all realised values of a random variable will fall within a given interval, in other words that the probability will be 1, but nevertheless there is a “high probability”. In the case of the Strong Law, however, we <i>can</i> say with probability 1 that the realised values will fall within the given interval because we are able to say that the values converge <i>almost surely.</i></div>
<div><i><br/></i></div>
<div><a href="https://www.youtube.com/watch?v=Bn0wWZENeQI">https://www.youtube.com/watch?v=Bn0wWZENeQI</a></div>
<div><br/></div>
<div>Weak Law:</div>
<div><br/></div>
<div><span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;"><span style=""><span style=" clip: rect(3.186em, 1001.33em, 4.167em, -1000em); top: -4.018em; left: 0.165em;">( lim </span><span style=" clip: rect(3.552em, 1001.61em, 4.273em, -1000em); top: -3.343em; left: 0em;"><span style="font-style: italic;">n</span>→∞</span></span></span></span></span></span></span></span></span></span><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;"><span style="padding-left: 0.188em;">)</span></span></span></span></span></span></span></span></span><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;"><i><span style="padding-left: 0.188em;">Pr</span></i>( |</span></span></span></span></span></span></span></span></span><i>x̅</i> <span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;">-<span style="font-style: italic; padding-left: 0.25em;">μ</span>|<span style="padding-left: 0.313em;">≥</span><span style="font-style: italic; padding-left: 0.313em;">ϵ </span>)<span style="padding-left: 0.313em;">=</span><span style="padding-left: 0.313em;">0</span></span></span></span></span></span></span></span></span></span></div>
<div><br/></div>
<div>Strong Law:</div>
<div><br/></div>
<div>Pr<span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;"><span style=""><span style=" clip: rect(3.186em, 1001.33em, 4.167em, -1000em); top: -4.018em; left: 0.165em;">( lim </span><span style=" clip: rect(3.552em, 1001.61em, 4.273em, -1000em); top: -3.343em; left: 0em;"><span style="font-style: italic;">n</span>→∞ </span></span></span></span></span></span></span></span></span></span></span><i>x̅ </i>= <i>E</i>( <i>x </i>)) = 1</div>
<div><br/></div>
<div><a href="https://terrytao.wordpress.com/2008/06/18/the-strong-law-of-large-numbers/">https://terrytao.wordpress.com/2008/06/18/the-strong-law-of-large-numbers/</a></div>
<div><br/></div>
<div>It is important to note that for the Law of Large Numbers to apply, the size of the population does not matter. The number <i>n</i> that should be sampled out of a population so that there is at most a chance 𝛿 that the estimate is off by more than <img src="Foundations%20of%20Data%20Science.resources/15E66256-BC22-4FF1-8E90-A54442C53BE6.svg" height="0" width="0"/> depends only on 𝛿 <i>and </i><img src="Foundations%20of%20Data%20Science.resources/15E66256-BC22-4FF1-8E90-A54442C53BE6.svg" height="0" width="0"/> and not on the overall population size.</div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: How do you Prove the Law of Large Numbers?</i></span></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>You can prove it by using Chebyshev’s inequality, which in turn relies on Markov’s inequality.</div>
<div><br/></div>
<div>Proof of Markov’s inequality: <a href="https://www.youtube.com/watch?v=sp9RF0zH-SU">https://www.youtube.com/watch?v=sp9RF0zH-SU</a></div>
<div><br/></div>
<div>Proof of Chebyshev’s inequality: <a href="https://www.youtube.com/watch?v=h0YH79kLuOA">https://www.youtube.com/watch?v=h0YH79kLuOA</a></div>
<div><br/></div>
<div>Proof of the Law of Large Numbers: <a href="https://www.youtube.com/watch?v=4QAeBJVn9WI">https://www.youtube.com/watch?v=4QAeBJVn9WI</a></div>
<div><br/></div>
<div>Courtesy of Ben Lambert’s excellent Youtube channel: <a href="https://www.youtube.com/channel/UC3tFZR3eL1bDY8CqZDOQh-w">https://www.youtube.com/channel/UC3tFZR3eL1bDY8CqZDOQh-w</a></div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: What is the Central Limit Theorem?</i></span></div>
<div><br/></div>
<div>The Central Limit Theorem is concerned with the sampling distribution of the mean.</div>
<div><br/></div>
<div>"The central limit theorem states that if you have a population with mean <i>μ</i> and standard deviation <i>σ</i> and take sufficiently large random samples from the population <a>with replacement</a><img src="Foundations%20of%20Data%20Science.resources/14484E1B-664F-4C46-ADA1-10AF3EB57C71.gif" height="1" width="1"/>, then the distribution of the sample means will be approximately normally distributed. This will hold true regardless of whether the source population is normal or skewed, provided the sample size is sufficiently large (usually n <u>&gt;</u> 30). If the population is normal, then the theorem holds true even for samples smaller than 30."</div>
<div><br/></div>
<div><a href="http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Probability/BS704_Probability12.html">http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Probability/BS704_Probability12.html</a></div>
<div><br/></div>
<div>Suppose the mean of the sample is <i>x̅</i></div>
<div>The size of the sample is <i>n</i></div>
<div>The standard deviation of the sample is <i>s</i></div>
<div>There are 4 aspects:</div>
<ol>
<li>the sampling distribution of the mean will be less spread than the values in the population</li>
<li>the sampling distribution will be well modelled by a normal distribution</li>
<li>the spread of the sampling distribution is related to the spread of the population values</li>
<li>bigger samples lead to a smaller spread in the sampling distribution</li>
</ol>
<div><br/></div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: How does the Law of Large Numbers reflect in high dimensions?</i></span></div>
<div><br/></div>
<div>"The square of the distance between two points <i>y</i> and <i>z</i> can be viewed as the sum of <i>d</i> independent samples of a random variable <i>x</i> that is the <i>squared difference of two Gaussians</i>."</div>
<div><br/></div>
<div style="text-align: center">| <b>y</b> - <b>z </b>|² = ( <i>i</i>=1∑<i>d </i>) ( <i>yᵢ - zᵢ </i>) ²</div>
<div><br/></div>
<div>Based on the Law of Large Numbers we can say that, with high probability, the sum is close to the sum’s expectation. </div>
<div><br/></div>
<div><i><span style="font-size: 18px;">Question: What is the difference between the Probability Density Function and Probability Mass Function?</span></i></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>Probability distribution can occur in two ways, depending on the characteristics of the random variable. If the random variable only has discrete values, the probability distribution is a Probability Mass Function (PMF) and the solution can be calculated as a weighted sum. If the random variable has continuous values, the probability is a Probability Density Function (PDF) and the solution can be calculated using an integral.</div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div><b>PMF</b> - Probability mass function relates to the use of discrete random variables</div>
<div><br/></div>
<div><span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(1.698em, 1016.22em, 2.856em, -1000em); top: -2.53em; left: 0em;"><span style=""><span style=" top: -3.868em; left: 0.611em;">Pr( <i>X</i></span></span>( <span style=""><span style=" clip: rect(3.428em, 1000.45em, 4.178em, -1000em); top: -4.018em; left: 0em;"><span style="font-style: italic;">xᵢ </span></span></span>) )<span style="padding-left: 0.313em;">=</span><span style="padding-left: 0.313em;">Pr</span>( <span style="font-style: italic;">X</span><span style="padding-left: 0.313em;">=</span><span style="padding-left: 0.313em;"><span style=""><span style=" clip: rect(3.428em, 1000.45em, 4.178em, -1000em); top: -4.018em; left: 0em;"><span style="font-style: italic;">xᵢ</span></span></span></span> ),<span style="padding-left: 0.188em;"> for </span><span style="font-style: italic;">i</span><span style="padding-left: 0.313em;">=</span><span style="padding-left: 0.313em;">1</span>,<span style="padding-left: 0.188em;">2</span>,<span style="padding-left: 0.188em;">3</span>,<span style="padding-left: 0.188em;">.</span><span style="padding-left: 0.188em;">.</span><span style="padding-left: 0.188em;">.</span><span style="padding-left: 0.188em;">,</span></span></span></span></span></div>
<div><br/></div>
<div>∑<span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(1.698em, 1016.22em, 2.856em, -1000em); top: -2.53em; left: 0em;"><span style=""><span style=" top: -3.868em; left: 0.611em;">Pr( <i>X</i></span></span>(<span style=""><span style=" clip: rect(3.428em, 1000.45em, 4.178em, -1000em); top: -4.018em; left: 0em;"><span style="font-style: italic;">xᵢ </span></span></span>) )</span></span></span></span> <span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(1.698em, 1016.22em, 2.856em, -1000em); top: -2.53em; left: 0em;"><span style=""><span style=" clip: rect(3.428em, 1000.45em, 4.178em, -1000em); top: -4.018em; left: 0em;">= 1</span></span></span></span></span></span></div>
<div><span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(1.698em, 1016.22em, 2.856em, -1000em); top: -2.53em; left: 0em;"><span style=""><span style=" top: -3.868em; left: 0.611em;">Pr( </span></span><span style=""><span style=" clip: rect(3.428em, 1000.45em, 4.178em, -1000em); top: -4.018em; left: 0em;"><span style="font-style: italic;">xᵢ </span></span></span>) &gt; 0</span></span></span></span></div>
<div><span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(1.698em, 1016.22em, 2.856em, -1000em); top: -2.53em; left: 0em;"><span style=""><span style=" top: -3.868em; left: 0.611em;">Pr( x ) = 0 for all other x</span></span></span></span></span></span></div>
<div><span style="font-family: STIXGeneral;"><br/></span></div>
<div><b>PDF</b> - Probability density function relates to the use of continuous random variables</div>
<div><br/></div>
<div><span style="text-align: center;"><span style="clip: rect(1.021em, 1023.96em, 3.378em, -1000em); top: -2.53em; left: 0em;"><span style="font-family: Arial;"><span style=" clip: rect(3.191em, 1000.42em, 4.374em, -1000em); top: -4.018em; left: 0em;"><span style="font-style: italic; text-rendering: optimizeLegibility;">f</span></span><span style=" top: -3.868em; left: 0.278em;"><span style="font-style: italic;">X</span></span></span><font face="Arial">( </font><span style="font-family: Arial; font-style: italic;">x </span><font face="Arial">)</font><span style="font-family: Arial; padding-left: 0.313em;">=</span><span style="font-family: Arial; padding-left: 0.313em;"><span style=""><span style=" clip: rect(3.186em, 1002.74em, 4.344em, -1000em); top: -4.694em; left: 50%;"><span style="font-style: italic;">d</span><span style=""><span style=" clip: rect(3.216em, 1000.65em, 4.167em, -1000em); top: -4.018em; left: 0em;"><span style="font-style: italic;">F</span></span><span style=" top: -3.868em; left: 0.611em;"><span style="font-style: italic;">X</span></span></span>( <span style="font-style: italic;">x </span>)</span><span style=" clip: rect(3.186em, 1000.97em, 4.18em, -1000em); top: -3.332em; left: 50%;"><span style="font-style: italic;">d</span><span style="font-style: italic;">x</span></span></span></span><span style="font-family: Arial; padding-left: 0.313em;">=</span><span style="padding-left: 0.313em;"><span style="font-family: Arial; clip: rect(3.216em, 1000.65em, 4.167em, -1000em); top: -4.018em; left: 0em;"><i>F </i></span><span style="clip: rect(3.407em, 1000.54em, 4.167em, -1000em); top: -3.721em; left: 0.611em;"><font face="Arial">‘<i>X</i></font></span></span><font face="Arial">( </font><span style="font-family: Arial; font-style: italic;">x </span><font face="Arial">),   </font><span style="font-family: Arial; padding-left: 0.188em;">if </span><span style="font-family: Arial;"><span style=" clip: rect(3.216em, 1000.65em, 4.167em, -1000em); top: -4.018em; left: 0em;"><span style="font-style: italic;">F</span></span><span style=" top: -3.868em; left: 0.611em;"><span style="font-style: italic;">X</span></span></span><font face="Arial">( </font><span style="font-family: Arial; font-style: italic;">x </span><font face="Arial">) is differentiable at </font><span style="font-family: Arial; font-style: italic;">x</span></span></span></div>
<div><i><span style="font-family: Arial;"><br/></span></i></div>
<div><span style="font-family: Arial;">Pr(<i>-</i></span><span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;"><span style=""><span style=" clip: rect(3.552em, 1001.61em, 4.273em, -1000em); top: -3.343em; left: 0em;">∞</span></span></span></span></span></span></span></span></span></span></span> <span style="font-family: Arial;">≤ <i>X</i> ≤</span> <span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;"><span style=""><span style=" clip: rect(3.552em, 1001.61em, 4.273em, -1000em); top: -3.343em; left: 0em;">∞</span></span></span></span></span></span></span></span></span></span></span><span style="font-family: Arial;">) = ₋</span><span style=""><span style=""><span style=" font-size: 111%;"><span style=" clip: rect(1.712em, 1002.34em, 2.952em, -1000em); top: -2.583em; left: 0em;"><span style=""><span style=" top: -3.874em; left: 0.523em;"><span style="font-size: 70.7%; font-family: STIXGeneral;">∞</span></span></span></span></span></span></span><span style="font-family: Arial;">∫</span><span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;"><span style=""><span style=" clip: rect(3.552em, 1001.61em, 4.273em, -1000em); top: -3.343em; left: 0em;">∞</span></span></span></span></span></span></span></span></span></span></span> <i>fX( x )dx = 1</i></div>
<div><span style="font-family: Arial;"><br/></span></div>
<div><span style="font-family: Arial;">Pr( <i>a</i> ≤ <i>X</i> ≤ <i>b </i>) = ₐ∫</span>ᵇ <i>fX( x )dx</i></div>
<div><i><br/></i></div>
<div>A PDF in fact <i>describes </i>a continuous random variable. I.e. if a random variable takes values on a continuous set, that by itself isn’t enough to make it a continuous random variable.</div>
<div><br/></div>
<div>The probability that X is any <i>particular</i> point is in fact 0 (the integral where <i>a</i> is equal to <i>b,</i> is 0). A side effect of this is that the probability of a closed and open interval (one that does, and one that doesn’t include the endpoints) is identical.</div>
<div><br/></div>
<div><i>Note:</i> it is easy to confuse Probability with the Probability Density Function, however they are not the same. Probability density functions are not probabilities, but if <span style=""><span style=""><span style=" font-size: 111%;"><span style=" clip: rect(1.726em, 1001.75em, 2.95em, -1000em); top: -2.583em; left: 0em;"><span style="font-family: STIXGeneral; font-style: italic;">𝑓</span><span style="font-family: STIXGeneral;">(</span><span style="font-family: STIXGeneral; font-style: italic;">𝑥</span><span style="font-family: STIXGeneral;">)</span></span></span></span></span> is a probability density function, then <span style="font-size: 16px;"><span style="font-family: Helvetica;"><span style=""><span style=""><span style=" clip: rect(1.542em, 1006.57em, 3.233em, -1000em); top: -2.583em; left: 0em;">𝑃<span style="padding-left: 0.313em;">=</span></span></span></span>ᵪ₀ <span style=""><span style=""><span style=" clip: rect(1.542em, 1006.32em, 3.233em, -1000em); top: -2.583em; left: 0em;"><span style=""><span style=" clip: rect(3.022em, 1000.64em, 4.526em, -1000em); top: -4.024em; left: 0em;"><span style="text-rendering: optimizeLegibility; vertical-align: -0.002em;">∫</span></span></span></span></span></span>ᵡ¹<span style=""><span style=""><span style=" clip: rect(1.542em, 1006.57em, 3.233em, -1000em); top: -2.583em; left: 0em;"><span style="padding-left: 0.188em;">𝑓</span>(𝑥)𝑑𝑥</span></span></span></span></span> is a probability and thus <span style="font-size: 16px;"><span style="font-family: STIXGeneral;">ᵪ₀<span style=""><span style=""><span style=" clip: rect(1.542em, 1006.32em, 3.233em, -1000em); top: -2.583em; left: 0em;"><span style=""><span style=" clip: rect(3.022em, 1000.64em, 4.526em, -1000em); top: -4.024em; left: 0em;"><span style="text-rendering: optimizeLegibility; vertical-align: -0.002em;">∫</span></span></span></span></span></span>ᵡ¹</span></span><span style=""><span style=""><span style=" font-size: 111%;"><span style=" clip: rect(1.542em, 1006.32em, 3.233em, -1000em); top: -2.583em; left: 0em;"><span style="font-family: STIXGeneral; font-style: italic; padding-left: 0.188em;">𝑓</span><span style="font-family: STIXGeneral;">(</span><span style="font-family: STIXGeneral; font-style: italic;">𝑥</span><span style="font-family: STIXGeneral;">)</span><span style="font-family: STIXGeneral; font-style: italic;">𝑑</span><span style="font-family: STIXGeneral; font-style: italic;">𝑥</span><span style="font-family: STIXGeneral; padding-left: 0.313em;">≤</span><span style="font-family: STIXGeneral; padding-left: 0.313em;">1</span></span></span></span></span> for all <span style=""><span style=""><span style=" font-size: 111%;"><span style=" clip: rect(1.961em, 1002.4em, 2.923em, -1000em); top: -2.583em; left: 0em;"><span style=""><span style=" clip: rect(3.403em, 1000.51em, 4.213em, -1000em); top: -4.024em; left: 0em;"><span style="font-family: STIXGeneral; font-style: italic;">𝑥</span></span><span style=" top: -3.874em; left: 0.55em;"><span style="font-size: 70.7%; font-family: STIXGeneral;">0</span></span></span><span style="font-family: STIXGeneral;">,</span><span style="padding-left: 0.188em;"><span style=""><span style=" clip: rect(3.403em, 1000.51em, 4.213em, -1000em); top: -4.024em; left: 0em;"><span style="font-family: STIXGeneral; font-style: italic;">𝑥</span></span><span style=" top: -3.874em; left: 0.55em;"><span style="font-size: 70.7%; font-family: STIXGeneral;">1</span></span></span></span></span></span></span></span> (<span style=""><span style=""><span style=" font-size: 111%;"><span style=" clip: rect(1.793em, 1003.24em, 2.923em, -1000em); top: -2.583em; left: 0em;"><span style=""><span style=" clip: rect(3.403em, 1000.51em, 4.213em, -1000em); top: -4.024em; left: 0em;"><span style="font-family: STIXGeneral; font-style: italic;">𝑥</span></span><span style=" top: -3.874em; left: 0.55em;"><span style="font-size: 70.7%; font-family: STIXGeneral;">0</span></span></span><span style="font-family: STIXGeneral; padding-left: 0.313em;">≤</span><span style="padding-left: 0.313em;"><span style=""><span style=" clip: rect(3.403em, 1000.51em, 4.213em, -1000em); top: -4.024em; left: 0em;"><span style="font-family: STIXGeneral; font-style: italic;">𝑥</span></span><span style=" top: -3.874em; left: 0.55em;"><span style="font-size: 70.7%; font-family: STIXGeneral;">1</span></span></span></span></span></span></span></span>).</div>
<div><br/></div>
<div>The general form of the PDF for the normal distribution is:</div>
<div><br/></div>
<div style="text-align: center"><img src="Foundations%20of%20Data%20Science.resources/Probability%20Density%20Function%203.png" height="112" width="456"/></div>
<div><br/></div>
<div>In the standard normal distribution <i>µ</i> = 0 and <i>σ</i> = 1, so that gives the simpler version:</div>
<div><br/></div>
<div style="text-align: center"><img src="Foundations%20of%20Data%20Science.resources/Probability%20Density%20Function%201.png" height="114" width="308"/></div>
<div style="text-align: center"><br/></div>
<div><a href="https://www.youtube.com/watch?v=8QFpZ3FndBc">https://www.youtube.com/watch?v=8QFpZ3FndBc</a></div>
<div><i><br/></i></div>
<div>To prove that there is a vanishingly small probability that a randomly generated point <b>z</b><i> </i>in <i>d</i>-dimensions would lie in the unit ball, Blum et al use a PDF with variance set at 1/2<i>π</i> so that the Gaussian probability density equals one at the origin (p. 7). This relationship is explained over here:</div>
<div><br/></div>
<div><a href="https://books.google.co.uk/books?id=AKuMj4PN_EMC&amp;pg=PA131&amp;lpg=PA131&amp;dq=probability+%22density+at+the+origin%22&amp;source=bl&amp;ots=EMqkf67xBd&amp;sig=ACfU3U0TRJIjhaLxZlpSY2eEOxz5au52JA&amp;hl=en&amp;sa=X&amp;ved=2ahUKEwiQ4I64k4bqAhUTrHEKHakYBM0Q6AEwB3oECAgQAQ#v=onepage&amp;q=probability%20%22density%20at%20the%20origin%22&amp;f=false">https://books.google.co.uk/books?id=AKuMj4PN_EMC&amp;pg=PA131&amp;lpg=PA131&amp;dq=probability+%22density+at+the+origin%22&amp;source=bl&amp;ots=EMqkf67xBd&amp;sig=ACfU3U0TRJIjhaLxZlpSY2eEOxz5au52JA&amp;hl=en&amp;sa=X&amp;ved=2ahUKEwiQ4I64k4bqAhUTrHEKHakYBM0Q6AEwB3oECAgQAQ#v=onepage&amp;q=probability%20%22density%20at%20the%20origin%22&amp;f=false</a></div>
<div><br/></div>
<div>So, given the formula:</div>
<div><br/></div>
<div>Pr( <i>x</i> = 0 ) = 1/( 2<i>πσ² </i>)<i>ᴺᐟ²</i></div>
<div><br/></div>
<div>if we set the variance to 1/2<i>π</i> then the probability at x = 0 becomes 1 because the function becomes 1<i>ᴺᐟ² ,</i> and 1 to the power of anything remains 1.</div>
<div><br/></div>
<div>The argument goes:</div>
<div>- there is vanishingly small probability that a random point <b>z</b><i> </i>in <i>d</i>-dimensions would lie in the unit ball</div>
<div>- this implies the integral of the probability density over the unit ball is vanishingly small</div>
<div>- and because the probability density in the unit ball is bounded below by a constant ...</div>
<div>- the unit ball must have vanishingly small volume</div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: What is a tail bound and how do we calculate its probability?</i></span></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>A tail bound is the bound probability of rare events. There are many ways to calculate tail bound probabilities. There are a number of famous theorems that can calculate bounds to various levels of precision, depending on the type of information available.</div>
<div><br/></div>
<div>These include the Markov, Chebyshev and Chernoff inequalities, as well as others like the Higher Moments, Gaussian Annulus, and Power Law theorems.</div>
<div><br/></div>
<div><a href="https://courses.cs.washington.edu/courses/cse312/11au/slides/09tails.pdf">https://courses.cs.washington.edu/courses/cse312/11au/slides/09tails.pdf</a></div>
<div><br/></div>
<div>There is a more general version called the <i>Master Tail Bounds Theorem</i> from which some of these theorems can be derived.</div>
<div><br/></div>
<div>In summary, the theorem says that given:</div>
<div><br/></div>
<div><i>x</i> = <i>x₁</i> + <i>x₂</i> + … + <i>x</i><span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; font-family: &quot;Helvetica Neue&quot;; -webkit-font-kerning: none; color: rgb(0, 0, 0);"><sub><span style="font-family: Arial;"><span style="font-size: 7.3px;"><i><span style="font-size: 10px;">n</span></i></span> <span style="font-size: 14px;"> </span></span></sub></span>   , and <i>x<span style="font-size: 10px;"><span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; font-family: &quot;Helvetica Neue&quot;; -webkit-font-kerning: none; color: rgb(0, 0, 0);"><i><sub><span style="font-size: 10px;"><span style="font-family: Arial;">i</span></span></sub></i></span></span></i> is iid</div>
<div>Variance ≤ <i>σ</i><sup>2</sup></div>
<div><sup><span style="font-size: 14px;">Zero mean</span></sup></div>
<div>0 ≤ <i>a</i> ≤ √2 . <i>nσ</i><sup>2</sup></div>
<div>| <i>E</i>( <i>x<span style="font-size: 10px;"><span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; font-family: &quot;Helvetica Neue&quot;; -webkit-font-kerning: none; color: rgb(0, 0, 0);"><i><sub><span style="font-size: 10px;"><span style="font-family: Arial;">i</span></span></sub></i></span></span></i>^<i>s </i>) | ≤ <i>σ</i><sup>2</sup><i>s</i>!    , for <i>s</i> = 3, 4 … [ ( <i>a<sup>2</sup></i>/4<i>nσ<sup>3 </sup></i>) ]</div>
<div><br/></div>
<div>Then:</div>
<div style="text-align: center">Pr( | <i>x </i>| ≥ <i>a </i>) ≤ 3<i>e</i> ^( -<i>a<sup>2 </sup></i>/ ( 12<i>nσ</i><sup>2 </sup>) )</div>
<div><br/></div>
<div>It gives a much stronger bound with respect to <i>a</i> than Markov and Chebyshev, for example. Here we have an exponential drop off with respect to a, whereas Markov gives us 1/<i>a</i> and Chebyshev gives us 1/<i>a<sup>2</sup></i>.</div>
<div><br/></div>
<div><a href="https://medium.com/jun-devpblog/data-science-1-expectation-variance-law-of-large-numbers-2ff49caf8b7d">https://medium.com/jun-devpblog/data-science-1-expectation-variance-law-of-large-numbers-2ff49caf8b7d</a></div>
<div><br/></div>
<div><i><span style="font-size: 18px;">Question: How is volume distributed in higher dimensions?</span></i></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>Volume in higher dimensions is distributed near the surface.</div>
<div><br/></div>
<div>Consider an object <i>A</i> in <i>R<sup>d</sup></i> shrunk by a small amount <i>𝛜</i>. The new object is represented by:</div>
<div><br/></div>
<div style="text-align: center">( 1 - <i>𝛜 </i>) <i>A =</i> { ( 1 - <i>𝛜</i> ) <i>x</i> | <i>x</i> ∈ <i>A</i> }</div>
<div><br/></div>
<div>This means that:</div>
<div><br/></div>
<div style="text-align: center">volume( ( 1 - <i>𝛜</i> ) <i>A</i> ) = ( 1 - <i>𝛜</i> )<i><sup>d </sup></i>volume( <i>A</i> )</div>
<div><br/></div>
<div>Because 1 - <i>x</i> ≤ <i>e<sup>-x</sup></i>, for any given object <i>A</i> in <i>R<sup>d</sup></i> we get:</div>
<div><br/></div>
<div style="text-align: center">volume( ( 1 - <i>𝛜</i> ) <i>A</i> ) / volume( <i>A</i> ) = ( 1 - <i>𝛜</i> )<sup><i>d</i></sup> ≤ <i>e^-</i><i>𝛜 d</i></div>
<div><i><br/></i></div>
<div>If we fix<i> 𝛜</i> and <i>d</i> ⟶ ∞ then the above approaches 0. Thus the volume of <i>A</i> is in the portion not represented by  ( 1 - <i>𝛜</i> ) <i>A.</i></div>
<div><br/></div>
<div>Given a unit ball <i>S</i> in <i>d</i>-dimensions the implication of the above is that at least 1 - <i>e^-𝛜 d</i> of the volume of <i>S</i> is concentrated in a <i>d</i>-dimensional annulus of width <i>𝛜</i> at the perimeter. We can see that <i>e^-𝛜 d</i> decreases rapidly as <i>d</i> increases, eg. for 𝛜 = 0.1, <i>d</i> = 3, e^-ed = 0.741, and for 𝛜 = 0.1, d = 100, e^-ed = 0.000045. So 1 - <i>e^-𝛜 d </i>⟶ 1 as <i>d </i>⟶ ∞.</div>
<div><br/></div>
<div>Another way of looking at it is to say that the unit ball’s volume is concentrated in a <i>d</i>-dimensional annulus of width <i>O</i>( 1/<i>d</i> ) or (1 - 1/<i>d</i>)<i> </i>near the perimeter. Given a ball of radius <i>r</i>, the width where the volume is concentrated is <i>O</i>( <i>r</i>/<i>d </i>).</div>
<div><br/></div>
<div>Q: Other than the intuition that large <i>d</i> accounts for the relationship between volume and the width of the annulus, how is the leap made, exactly, to the annulus’ width <i>O</i>( 1/<i>d</i> )?</div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: How do you calculate the volume of the unit ball in d dimensions?</i></span></div>
<div><br/></div>
<div>As <i>d</i> ⟶ ∞ , the volume of the ball goes to zero. In the first few dimensions the volume goes up, up to d = 7, and then it goes down.</div>
<div><br/></div>
<div>The volume can be calculated using integration. There is more than one way. Integration in Cartesian coordinates have complicated integral limits, so polar coordinates are preferred.</div>
<div><br/></div>
<div>Let <i>V</i> ( <i>d</i> ) is the volume of the unit ball, and <i>A</i> ( <i>d</i> ) is the surface. The proof starts with:</div>
<div><br/></div>
<div style="text-align: center"><span style="font-family: Arial;">V( <i>d</i> ) =</span> <span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;"><span style=""><span style=" clip: rect(3.552em, 1001.61em, 4.273em, -1000em); top: -3.343em; left: 0em;"><span style="font-size: 10px;"><span style="font-family: STIXGeneral;">S^d</span></span></span></span></span></span></span></span></span></span></span></span></span><span style="font-family: Arial;">∫ </span><span style=""><span style=""><span style=" font-size: 111%;"><span style=" clip: rect(1.712em, 1002.34em, 2.952em, -1000em); top: -2.583em; left: 0em;"><span style=""><span style=" top: -3.874em; left: 0.523em;"><span style="font-size: 70.7%; font-family: STIXGeneral;">r=0</span></span></span></span></span></span></span><span style="font-family: Arial;">∫¹</span> r^d-1 <i>drdΩ</i></div>
<div style="text-align: center"><i><br/></i></div>
<div style="text-align: center"><span style="font-family: Arial;">V( <i>d</i> ) =</span> <span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;"><span style=""><span style=" clip: rect(3.552em, 1001.61em, 4.273em, -1000em); top: -3.343em; left: 0em;"><span style="font-size: 10px;"><span style="font-family: STIXGeneral;">S^d</span></span></span></span></span></span></span></span></span></span></span></span></span><span style="font-family: Arial;">∫</span> <i>dΩ </i><span style=""><span style=""><span style=" font-size: 111%;"><span style=" clip: rect(1.712em, 1002.34em, 2.952em, -1000em); top: -2.583em; left: 0em;"><span style=""><span style=" top: -3.874em; left: 0.523em;"><span style="font-size: 70.7%; font-family: STIXGeneral;">r=0</span></span></span></span></span></span></span><span style="font-family: Arial;">∫¹</span> r^d-1 <i>dr</i></div>
<div style="text-align: center"><i><br/></i></div>
<div style="text-align: center"><i>= 1/d </i><span style="font-family: Arial;"><span style="text-align: center;"><span style=""><span style=" clip: rect(0.93em, 1010.13em, 3.034em, -1000em); top: -2.232em; left: 0em;"><span style="padding-right: 0.167em; padding-left: 0.167em;"><span style=""><span style=" clip: rect(2.716em, 1009.96em, 4.82em, -1000em); top: -4.018em; left: 0em;"><span style=""><span style=" clip: rect(2.844em, 1009.96em, 4.948em, -1000em); top: -4.146em; right: 0em;"><span style=""><span style=" clip: rect(3.552em, 1001.61em, 4.273em, -1000em); top: -3.343em; left: 0em;"><span style="font-size: 10px;"><span style="font-family: STIXGeneral;">S^d</span></span></span></span></span></span></span></span></span></span></span></span></span><span style="font-family: Arial;">∫</span> <i>dΩ</i></div>
<div style="text-align: center"><br/></div>
<div style="text-align: center"><i>= A ( d ) / d</i></div>
<div style="text-align: center"><i><br/></i></div>
<div>But this stops integration at the surface of the sphere. To allow it to go all the way to infinity, involve an exponential in a function called <i>I</i> ( <i>d</i> ). <i>I</i> ( <i>d</i> ) can then be calculated in both Cartesian and polar coordinates, which yields:</div>
<div><br/></div>
<div style="text-align: center"><i>I</i> ( <i>d</i> ) = <i>π</i>^<i>d</i>/2</div>
<div style="text-align: center"><br/></div>
<div>and</div>
<div><br/></div>
<div style="text-align: center"><i>I</i> ( <i>d</i> ) = <i>A</i> ( <i>d</i> ) 1/2 <span style="font-family: Arial;">Γ( <i>d</i> / 2 )</span></div>
<div style="text-align: center"><span style="font-family: Arial;"><br/></span></div>
<div><span style="font-family: Arial;">which together gives:</span></div>
<div><span style="font-family: Arial;"><br/></span></div>
<div style="text-align: center"><span style="font-family: Arial;"><i>A</i>( <i>d</i> ) = <i>π</i>^( <i>d</i> / 2 ) / ( (1/2) Γ( <i>d</i> / 2 ) )</span></div>
<div><br/></div>
<div><span style="font-family: Arial;">That produces the lemma for the surface area <i>A</i>( <i>d</i> ) and the volume <i>V</i>( <i>d</i> ):</span></div>
<div><span style="font-family: Arial;"><br/></span></div>
<div style="text-align: center"><span style="font-family: Arial;"><i>A</i>( <i>d</i> ) = 2<i>π</i>^( <i>d</i> / 2 ) / Γ( <i>d</i> / 2 ) </span></div>
<div style="text-align: center"><span style="font-family: Arial;"><br/></span></div>
<div><span style="font-family: Arial;">and</span></div>
<div><span style="font-family: Arial;"><br/></span></div>
<div style="text-align: center"><span style="font-family: Arial;"><i>V</i>( <i>d</i> ) = 2<i>π</i>^( <i>d</i> / 2 ) / <i>d </i>Γ( <i>d</i> / 2 ) </span></div>
<div style="text-align: center"><span style="font-family: Arial;"><br/></span></div>
<div><span style="font-family: Arial;">Since </span><span style="font-family: Arial;"><i>π</i>^( <i>d</i> / 2 ) is an exponential in d / 2 and </span><span style="font-family: Arial;">Γ( <i>d</i> / 2 ) grows as the factorial of d / 2, the lim </span><i>d</i> ⟶ ∞ V( d ) = 0 (as claimed at the start).</div>
<div><br/></div>
<div>An intuitive explanation of the formula is available courtesy of Zach Star:</div>
<div><br/></div>
<div><a href="https://www.youtube.com/watch?v=mXp1VgFWbKc">https://www.youtube.com/watch?v=mXp1VgFWbKc</a></div>
<div><br/></div>
<div>Note that the above steps are an oversimplification of the proof. For a more general look at a proof for the formula of the volume of a ball in higher dimensions, see:</div>
<div><br/></div>
<div><a href="https://www.youtube.com/watch?v=XLq-cjwvS3M">https://www.youtube.com/watch?v=XLq-cjwvS3M</a></div>
<div><br/></div>
<div>Another example:</div>
<div><br/></div>
<div><a href="https://www.usna.edu/Users/physics/mungan/_files/documents/Scholarship/HypersphereVolume.pdf">https://www.usna.edu/Users/physics/mungan/_files/documents/Scholarship/HypersphereVolume.pdf</a></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div><i><span style="font-size: 18px;">Question: How is the volume of the unit ball distributed </span></i><i><span style="font-size: 18px;">in higher dimensions? </span></i></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>The claim is that in high dimensions most of a unit ball's volume is concentrated near its "equator”. This is tricky to understand and not easy to visualise, partly because the 3-dimensional version seems to be atypical. One way of thinking about is that, in actual fact, the volume is distributed evenly and that one lap around the ‘equator’ passes by an increasingly larger amount of its interior, and therefore volume, as <i>d</i> ⟶ ∞. In terms of a visualisation it helps to keep in mind that a unit ball becomes ’spiky’, however with some caveats as explained by Colin Wright:</div>
<div>
<div><br/></div>
<div><a href="http://www.penzba.co.uk/cgi-bin/PvsNP.py?SpikeySpheres">http://www.penzba.co.uk/cgi-bin/PvsNP.py?SpikeySpheres</a></div>
<div><br/></div>
</div>
<div><a href="https://news.ycombinator.com/item?id=3995615">https://news.ycombinator.com/item?id=3995615</a></div>
<div><br/></div>
<div>
<div>Blum et al puts it as follows: “most of the volume of the unit ball lies in the thin slab of points whose dot product with <b>v</b> has magnitude <i>O</i>( 1 / √<i>d</i> )”, which can be shown by fixing <b>v</b> as the first coordinate vector. From this it can be shown that, with high probability, two random points in the unit ball are nearly orthogonal. Specifically their vectors will be nearly orthogonal, they will be close to the surface, and have length 1 - <i>O</i>( 1 / <i>d</i> ). Fixing the first as “north”, the second will have a projection of ±<i>O</i>( 1 / √<i>d</i> ), which is the same as their dot product. This means that with high probability the angle between the vectors will be <i>π</i> / 2 ± <i>O</i>( 1 / √<i>d</i> ).</div>
<div><br/></div>
<div><b>So to summarise:</b></div>
</div>
<ol>
<li>As <i>d</i> ⟶ ∞ , the volume of the ball goes to zero</li>
<li>A unit ball’s volume is concentrated around its ‘equator’, i.e.: | x₁ | = <i>O</i>( 1 / √<i>d</i> )                 </li>
<li>The vectors of two random points in the unit ball tend to be (nearly) orthogonal </li>
<li>The vectors tend to be close to the surface, with length 1 - <i>O</i>( 1 / <i>d</i> )</li>
<li>The angles between the vectors tend to be <i>π</i> / 2 ± <i>O</i>( 1 / √<i>d</i> )</li>
</ol>
<div><br/></div>
<div>Here is an attempt to explain it intuitively:</div>
<div><br/></div>
<div><a href="https://mathoverflow.net/questions/210291/how-to-explain-the-concentration-of-measure-phenomenon-intuitively">https://mathoverflow.net/questions/210291/how-to-explain-the-concentration-of-measure-phenomenon-intuitively</a></div>
<div><br/></div>
<div>But a better way to visualise it actually comes from this explanation:</div>
<div><br/></div>
<div><a href="https://www.quora.com/Why-is-the-higher-the-dimension-the-less-the-hypervolume-of-a-hypersphere-inscribed-in-a-hypercube-occupy-the-hypervolume-of-the-hypercube">https://www.quora.com/Why-is-the-higher-the-dimension-the-less-the-hypervolume-of-a-hypersphere-inscribed-in-a-hypercube-occupy-the-hypervolume-of-the-hypercube</a></div>
<div><br/></div>
<div><br/></div>
<div>The distribution trend can be seen in these Python generated graphs:</div>
<div><br/></div>
<div><img src="Foundations%20of%20Data%20Science.resources/FCA6E41A-729F-460D-9BDC-EB60D87D75D7.png" height="278" width="368"/></div>
<div><img src="Foundations%20of%20Data%20Science.resources/CAFC28EB-77E5-4461-BBA5-D325B8EC465C.png" height="278" width="368"/></div>
<div><img src="Foundations%20of%20Data%20Science.resources/C1B7F4AC-DDBF-49BA-BAF5-8D712B086871.png" height="278" width="368"/></div>
<div><img src="Foundations%20of%20Data%20Science.resources/F7B911F4-1B33-4578-9E0A-E23DD28AB419.png" height="278" width="368"/></div>
<div><a href="https://www.johndcook.com/blog/2017/07/13/concentration_of_measure/">https://www.johndcook.com/blog/2017/07/13/concentration_of_measure/</a></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>The theorem for volume near the equator of the unit ball goes:</div>
<div><br/></div>
<div style="margin-left:40px;">For c ≥ 1 and d ≥ 3, at least a 1 - ( 2 / <i>c</i> ) <i>e</i>^( ( <i>c</i>^2 ) / 2 ) fraction of the volume of the d-dimensional unit ball has | <i>x</i>₁ | ≤ <i>c</i> / √( <i>d</i> - 1 ).</div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: What does it mean that two coordinates are nearly orthogonal?</i></span></div>
<div><br/></div>
<div>The theorem for near orthogonality goes:</div>
<div><br/></div>
<div>Consider drawing n points <i>x</i>₁, <i>x</i>₂, … , <i>x</i><span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; font-family: &quot;Helvetica Neue&quot;; -webkit-font-kerning: none; color: rgb(0, 0, 0);"><sub><span style="font-family: Arial;"><span style="font-size: 7.3px;"><i><span style="font-size: 10px;">n</span></i></span></span></sub></span> at random from the unit ball. With probability 1 - <i>O</i>( 1 / <i>n</i> )</div>
<div><br/></div>
<div>1. | <i>x</i>ᵢ | ≥ 1 - ( 2 ln <i>n </i>) / <i>n</i> for all <i>i</i>, and</div>
<div>2. | <i>x</i>ᵢ , <i>x</i>ⱼ | ≤ √( 6 ln <i>n </i>) /<i> </i>√( d - 1 ) for all <i>i ≠ j</i></div>
<div><i><br/></i></div>
<div>One way of measuring orthogonality is to calculate the squared dot product of a vector and other coordinate vectors throughout the unit ball. So for example if a random vector = (1, 0, 0, … 0), all these dot products give zero mean, with variance = 1 / <i>d</i>, and standard deviation = √( 1 / <i>d</i> ). To put it differently, the expected value of any two coordinate vectors' dot products is 0, because each component of the sum is the product of two independent Gaussians with mean 0. So we can conclude that in higher dimensions the inner dot product of any two coordinate vectors is likely to be 0, with high probability.</div>
<div><i><br/></i></div>
<div><a href="https://math.stackexchange.com/questions/995623/why-are-randomly-drawn-vectors-nearly-perpendicular-in-high-dimensions">https://math.stackexchange.com/questions/995623/why-are-randomly-drawn-vectors-nearly-perpendicular-in-high-dimensions</a></div>
<div><br/></div>
<div><a href="https://math.stackexchange.com/questions/3059747/probability-of-two-random-points-being-orthogonal-in-higher-dimensional-unit-sph">https://math.stackexchange.com/questions/3059747/probability-of-two-random-points-being-orthogonal-in-higher-dimensional-unit-sph</a></div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: How do you generate points uniformly at random from a ball?</i></span></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>Generating points <i>x</i>₁, <i>x</i>₂, …, <i>x<span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; -webkit-font-kerning: none; color: rgb(0, 0, 0);"><span style="font-size: 10px;"><span style="font-family: Arial;"><sub>d </sub></span></span></span></i>each with coordinates an independent Gaussian variable:</div>
<div><i>    </i> <i><span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; -webkit-font-kerning: none; color: rgb(0, 0, 0);">using zero mean, unit variance Gaussian, </span></i><i><span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; -webkit-font-kerning: none; color: rgb(0, 0, 0);">i.e.</span></i></div>
<div><i><span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; -webkit-font-kerning: none; color: rgb(0, 0, 0);">     1 / ( </span></i>√2<i>π</i> ) exp( -x² / 2 ) on the real line</div>
<div><br/></div>
<div>Gives probability density of <i>x</i>:</div>
<div><br/></div>
<div style="text-align: center">Pr(<b>x</b>) = ( 1 / ( 2<i>π</i> )^( <i>d</i> / 2 ) ) <i>e</i>^-( ( <i>x</i>₁^2 + <i>x</i>₂^2 … +<i>x<span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; -webkit-font-kerning: none; color: rgb(0, 0, 0);"><span style="font-size: 10px;"><span style="font-family: Arial;"><sub>d</sub></span></span></span></i>^2 ) / 2 )</div>
<div style="text-align: center"><br/></div>
<div>with spherical symmetry.</div>
<div><br/></div>
<div>Normalising vector <b>x</b> = ( <i>x</i>₁, <i>x</i>₂, …, <i>x<span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; -webkit-font-kerning: none; color: rgb(0, 0, 0);"><span style="font-size: 10px;"><span style="font-family: Arial;"><sub>d</sub></span></span></span></i> ) to a unit vector <b>x</b> / | <b>x</b> |</div>
<div><br/></div>
<div>Gives a distribution that is uniform over the sphere’s surface</div>
<div><i>Note:</i> once normalised, the coordinates are no longer independent</div>
<div><br/></div>
<div>Generating a point <b>y</b> uniformly over the unit ball volume,</div>
<div>we must scale the surface point <b>x</b> / | <b>x</b> | by a scalar <i>p</i> ∈ [0, 1]      [2]</div>
<div><br/></div>
<div>Gives a point:</div>
<div><br/></div>
<div style="text-align: center">y = <i>p</i> ( <b>x</b> / | <b>x</b> | )</div>
<div><br/></div>
<div><i>Note:</i> The distribution of <i>p</i> as a function of <i>r</i> is not uniform across the ball. Instead the density of <i>p</i> at distance <i>r</i> is proportional to <i>r</i>^( <i>d</i> - 1 ) in <i>d</i> dimensions. Solving<span style="font-family: Arial;"> ᵣ₌₀∫</span><span style="font-size: 18px;">ʳ</span>⁼¹ <i>c</i>( ( <i>r ^ d -</i> 1 ) )<i>dr</i> = 1<i> </i>we see that<i> </i>we should set<i> c</i> = <i>d.</i> Another way of looking at it is knowing that the density at radius <i>r</i> is exactly <i>dr</i>^(<i>d</i> - 1) <i>V<span style="font-style: normal; font-variant-caps: normal; font-weight: normal; font-stretch: normal; -webkit-font-kerning: none; color: rgb(0, 0, 0);"><i><span style="font-size: 10px;"><span style="font-family: Arial;"><sub>d </sub></span></span></i></span></i>. So we would want to pick <i>p(r)</i> with density equal to <i>dr</i>^(<i>d </i>- 1) for r over [0,1].</div>
<div><br/></div>
<div><a href="https://stats.stackexchange.com/questions/85916/distribution-of-scalar-products-of-two-random-unit-vectors-in-d-dimensions">https://stats.stackexchange.com/questions/85916/distribution-of-scalar-products-of-two-random-unit-vectors-in-d-dimensions</a></div>
<div><br/></div>
<div>On a more practical note, there are a number of ways to sample from a unit ball. Rejection sampling is a common and intuitive approach, but it is not very efficient. Two of the most well known, much more efficient ways, are the Box-Muller transform and the inverse transform sampling.</div>
<div><br/></div>
<div>The Box-Muller transform takes two uniformly distributed random numbers and derives Gaussian distributed random numbers.</div>
<div><br/></div>
<div><a href="https://www.youtube.com/watch?v=EXsdT91XFAY">https://www.youtube.com/watch?v=EXsdT91XFAY</a></div>
<div><br/></div>
<div>Inverse transform sampling requires that the CDF is known, and entities have to be normalised. Most importantly it doesn’t generalise well to higher dimensional problems, due to difficulties in calculating a CDF. Unlike rejection sampling, however, it is 100% efficient.</div>
<div><br/></div>
<div><a href="https://www.youtube.com/watch?v=rnBbYsysPaU">https://www.youtube.com/watch?v=rnBbYsysPaU</a></div>
<div><br/></div>
<div>This blog post discusses a variety of ways with some pseudo code that is fairly easy to implement.</div>
<div><br/></div>
<div><a href="http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/">http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/</a></div>
<div><br/></div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: How do Gaussians behave in Higher Dimensions?</i></span></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div>The d-dimensional spherical Gaussian with zero mean and <span style="font-family: &quot;Helvetica Neue&quot;;">variance</span> <span style="font-family: &quot;Helvetica Neue&quot;;"><span style="left: 583.982px; top: 952.295px;"><i>σ²</i> has the following density function:</span></span></div>
<div><i><span style="font-size: 18px;"><br/></span></i></div>
<div style="text-align: center">    <img src="Foundations%20of%20Data%20Science.resources/Screenshot%202020-07-01%20at%2010.39.05.png" height="126" width="530"/></div>
<div><span style="font-family: sans-serif;"><br/></span></div>
<div><span style="font-family: &quot;Helvetica Neue&quot;;">Although density is maximum at origin, there is little volume. The radius needs to be increased to around √d before there is significant and hence probability mass. Beyond √d the probability density starts to drop off at a much faster rate than the volume increases.</span></div>
<div><span style="font-family: &quot;Helvetica Neue&quot;;"><br/></span></div>
<div><span style="font-size: 18px;"><i><span style="font-family: &quot;Helvetica Neue&quot;;">Question: What is the connection between random projection and the Johnson-Lindenstrauss lemma?</span></i></span></div>
<div><span style="font-family: &quot;Helvetica Neue&quot;;"><br/></span></div>
<div><span style="font-size: 18px;"><i><span style="font-family: &quot;Helvetica Neue&quot;;">Question: How do you identify which Gaussian a point belongs to when there is more than one distribution?</span></i></span></div>
<div><i><span style="font-size: 18px;"><span style="font-family: &quot;Helvetica Neue&quot;;"><br/></span></span></i></div>
<div><span style="font-family: &quot;Helvetica Neue&quot;;">The algorithm is ultimately simple:</span></div>
<div><span style="font-family: &quot;Helvetica Neue&quot;;">- Calculate the distance between pairs of points</span></div>
<div><span style="font-family: &quot;Helvetica Neue&quot;;">- Points whose distance apart is smaller are from the same Gaussian (vs points further apart)</span></div>
<div><br/></div>
<div><font face="Helvetica Neue">Firstly, from the Gaussian Annulus Theorem, which states that for large <i>d</i>, the <i>d</i>-dimensional Gaussian is located in the annulus with high probability, we can derive that for two points <b>x</b> and <b>y</b>:</font></div>
<div><span style="font-family: &quot;Helvetica Neue&quot;;"><br/></span></div>
<div style="text-align: center"><font face="Helvetica Neue">|<b>x</b> - <b>y</b>| = √2<i>d</i> ± <i>O</i>(1)</font></div>
<div><br/></div>
<div>If we now have two Gaussians, with centres <b>p</b> and <b>q</b>, separated by distance ∆, then if point <b>x</b> is drawn from the first and point <b>y</b> from the second Gaussian, then the distance between them will be close to √( ∆² + 2<i>d</i> ) since <b>x</b> - <b>p</b>, <b>p</b> - <b>q</b>, <b>q</b> - <b>y</b> are mutually (nearly) perpendicular. It can then be show that:</div>
<div><br/></div>
<div style="text-align: center">|<b>x</b> - <b>y</b>|2 = ∆² + 2<i>d</i> <font face="Helvetica Neue">± <i>O</i>(√<i>d </i>)</font></div>
<div><br/></div>
<div>In order to ensure that two points picked from the same Gaussian are closer to each other than two points picked from different Gaussians requires that the upper limit of the distance between a pair of points from the same Gaussian is at most the lower limit of distance between point from different Gaussians.</div>
<div><br/></div>
<div>It can be derived from the above that spherical Gaussians can be separated this way as long as their centres are separated by <i>ω</i>(<i>d</i>¹ᐟ⁴).</div>
<div><br/></div>
<div>Refined algorithm:</div>
<div>- Calculate all pairwise distances between points</div>
<div>- The cluster of smallest pairwise distance must be from a single Gaussian</div>
<div>- Remove these points</div>
<div>- The remaining points come from the second Gaussian</div>
<div><br/></div>
<div><span style="font-size: 18px;"><i>Question: What is the difference between the Standard error and Standard deviation?</i></span></div>
<div><br/></div>
<div>Standard deviation quantifies the variation within a set of measurements. </div>
<div>Standard error quantifies the variation of the means of multiple sets of measurements.</div>
<div><br/></div>
<div><a href="https://www.youtube.com/watch?v=A82brFpdr9g">https://www.youtube.com/watch?v=A82brFpdr9g</a></div>
<div><br/></div>
</body></html>