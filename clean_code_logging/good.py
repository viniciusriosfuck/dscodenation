#!/usr/bin/env python

from math import sqrt


class CalculadoraGeometria:

    @classmethod
    def volume_piramide(cls, lado_base, altura):
        area_base = (lado_base ** 2 * sqrt(3)) / 4

        return (area_base * altura) / 3


def main():
    print(CalculadoraGeometria.volume_piramide(lado_base=3, altura=4))


if __name__ == '__main__':
    main()
