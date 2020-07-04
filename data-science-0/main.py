#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[ ]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[ ]:


def q1():
    return df.shape


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[ ]:


def q2():
    return df[(df.Gender == 'F') & (df.Age == '26-35')].count().User_ID.item()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[ ]:


def q3():
    return df.nunique().User_ID.item()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[ ]:


def q4():
    return df.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[ ]:


def q5():
    lines_na = df.isna().sum(axis=1) > 0
    return lines_na.mean().item()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[ ]:


def q6():
    return df.isna().sum().max().item()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:


def q7():
    return df.Product_Category_3.mode().values[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:


def q8():
    x = df.Purchase.to_numpy()
    x_mod = (x - x.min()) / (x.max() - x.min())
    return x_mod.mean().item()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variável `Purchase` após sua padronização? Responda como um único escalar.

# In[ ]:


def q9():
    x = df.Purchase.to_numpy()
    z = (x - x.mean()) / x.std()
    v_bool = (z >= -1.0) & (z <= 1.0)
    return v_bool.sum().item()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[ ]:


def q10():
    return True if df.Product_Category_3[df.Product_Category_2.isna()].isna().mean() == 1.0 else False

