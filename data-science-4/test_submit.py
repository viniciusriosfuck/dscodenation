from main import *


def test_q1():
    result = q1()

    assert len(result) == 11
    assert result == ['ASIA (EX. NEAR EAST)', 'BALTICS', 'C.W. OF IND. STATES', 'EASTERN EUROPE',
                        'LATIN AMER. & CARIB', 'NEAR EAST', 'NORTHERN AFRICA', 'NORTHERN AMERICA',
                        'OCEANIA', 'SUB-SAHARAN AFRICA', 'WESTERN EUROPE']

def test_q2():
    result = q2()

    assert result == 23

def test_q3():
    result = q3()

    assert result == 18

def test_q4():
    result = q4()

    assert result == -1.047

def test_q5():
    result = q5()

    assert len(result) == 3
    assert result[0] == 24
    assert result[1] == 26
    assert result[2] == False

def test_q6():
    result = q6()

    assert result == 213

def test_q7():
    result = q7()

    assert result == 8.888
