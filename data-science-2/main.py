#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# In[2]:


# %matplotlib inline

from IPython.core.pylabtools import figsize

from statsmodels.graphics.gofplots import qqplot



figsize(12, 8)

sns.set()


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[4]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[5]:


athletes.shape


# In[6]:


athletes.describe()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[10]:


height_sample_3000 = get_sample(df=athletes, col_name='height', n=3000)


# In[9]:


def check_normality(statistic, p_value, alpha=0.05):
    print('Statistics=%.3f, p_value=%.3f' % (statistic, p_value))
    if p_value <= alpha:
        seems_normal = False
        print('Sample does not look Gaussian (reject H0)')
    else:
        seems_normal = True
        print('Sample looks Gaussian (fail to reject H0)')
    return seems_normal
    


# In[11]:


def q1():
    statistic, p_value = sct.shapiro(height_sample_3000)
    return check_normality(statistic, p_value)


# In[12]:


# Test
q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[13]:


def plot_dist_qq_box(variable_to_plot, fit_legend='normal_fit'):
    fig, axes = plt.subplots(2, 2)
    l1 = sns.distplot(variable_to_plot, fit=sct.norm, kde=False, ax=axes[0,0])
    l2= sns.boxplot(variable_to_plot, orient='v' , ax=axes[0,1])
    l3 = qqplot(variable_to_plot, line='s', ax=axes[1,0])
    l4 = sns.distplot(variable_to_plot, fit=sct.norm,  hist=False, kde_kws={"shade": True}, ax=axes[1,1])
    axes[0,0].legend((fit_legend,'distribution'))
    axes[1,0].legend(('distribution',fit_legend))
    axes[1,1].legend((fit_legend,'kde_gaussian'));


# In[14]:


plot_dist_qq_box(height_sample_3000)


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[15]:


def q2():
    statistic, p_value = sct.jarque_bera(height_sample_3000)
    return check_normality(statistic, p_value)


# In[16]:


#Test
q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[17]:


weight_sample_3000 = get_sample(df=athletes, col_name='weight', n=3000)


# In[18]:


def q3():
    statistic, p_value = sct.normaltest(weight_sample_3000)
    return check_normality(statistic, p_value)


# In[19]:


#Test
q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[20]:


plot_dist_qq_box(weight_sample_3000)


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[21]:


log_weight_sample_3000 = np.log(weight_sample_3000)


# In[22]:


def q4():
    statistic, p_value = sct.normaltest(log_weight_sample_3000)
    return check_normality(statistic, p_value)


# In[23]:


#test
q4()


# In[24]:


plot_dist_qq_box(log_weight_sample_3000, fit_legend='lognormal_fit')


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# In[25]:


athletes.columns


# In[26]:


athletes['nationality'].value_counts()


# In[27]:


bra = athletes.loc[athletes['nationality']=='BRA']
bra.head()


# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[28]:


# bra = athletes.loc[athletes['nationality']=='BRA']
usa = athletes.loc[athletes['nationality']=='USA']
can = athletes.loc[athletes['nationality']=='CAN']


# In[29]:


def check_equal_means(statistic, p_value, alpha=0.05):
    print('Statistics=%.3f, p_value=%.3f' % (statistic, p_value))
    if p_value <= alpha/2:
        means_seems_equal = False
        print('Sample means not look equal (reject H0)')
    else:
        means_seems_equal = True
        print('Sample means look equal (fail to reject H0)')
    return means_seems_equal


# In[30]:


def q5():
    statistic, p_value = sct.ttest_ind(bra['height'].dropna(), usa['height'].dropna(), equal_var=False)
    return check_equal_means(statistic, p_value)


# In[31]:


# Teste
q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[32]:


def q6():
    statistic, p_value = sct.ttest_ind(bra['height'].dropna(), can['height'].dropna(), equal_var=False)
    return check_equal_means(statistic, p_value)


# In[33]:


# Teste
q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[34]:


def q7():
    statistic, p_value = sct.ttest_ind(usa['height'].dropna(), can['height'].dropna(), equal_var=False)
    check_equal_means(statistic, p_value)
    return p_value.round(8)


# In[35]:


# Teste
q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
