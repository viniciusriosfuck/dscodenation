#!/usr/bin/env python

import math

class Calculadora:
    def calcula(self, x, y):
        return (x**2 * math.sqrt(3) / 4) * y / 3

c = Calculadora()
print(c.calcula(3, 4))