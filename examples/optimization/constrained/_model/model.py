#!/ usr / bin / env python

#Minus g09

def model(k) : d = k["Parameters"] res = (d[0] - 10.0) * *2 + 5.0 * (d[1] - 12.0) * *2 + d[2] * *4 + 3.0 * (d[3] - 11.0) * *2 + 10.0 * d[4] * *6 + 7.0 * d[5] * *2 + d[6] * *4. - 4.0 * d[5] * d[6] - 10.0 * d[5] - 8.0 * d[6]

                                                                                                                                                                                                                      k["F(x)"] = -res
