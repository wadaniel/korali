#!/ usr / bin / env python
import math

#Minimum expected at - 0.5
  def evalmodel(s) : x = s["Parameters"][0] if (abs(x)> 3.0) : r = math.inf elif (abs(x)<0.5) : r = math.nan else : r = x * x + math.sin(x)

                                                                                    s["F(x)"] = - r
