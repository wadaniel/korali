# Distribution / Cauchy
   				   
##Description

The Cauchy distribution has the probability density function (PDF):

$$f(x;x_{0},\gamma )={\frac  {1}{\pi \gamma \left[1+\left({\frac  {x-x_{0}}{\gamma }}\right)^{2}\right]}}={1 \over \pi \gamma }\left[{\gamma ^{2} \over (x-x_{0})^{2}+\gamma ^{2}}\right],$$

where $x_{0}$ is the location parameter, specifying the **location** of the peak of the distribution, and $\gamma$ is the **scale** parameter which specifies the half-width at half-maximum (HWHM).

[Picture here?]

##Syntax

```python
  # Definition
  Korali["Problem"]["Variables"][i]["Distribution"] = "Cauchy"
  
  # Mandatory Settings
  Korali["Problem"]["Variables"][i]["Type"] = ...
  Korali["Problem"]["Variables"][i]["Location"] = ...
  Korali["Problem"]["Variables"][i]["Scale"] = ...
```

### **Location**
Indicates the location of the peak of the Cauchy distribution.

- Value Type: *Real Number*
- Example: 

```python
 Korali["Problem"]["Variables"][i]["Location"] = +8.0
```

### **Scale**
Specifies the half-width at half-maximum (HWHM) of the Cauchy distribution.

 - Value Type: *Real Number*
 - Example:

```python
Korali["Problem"]["Variables"][i]["Scale"] = +2.0
```