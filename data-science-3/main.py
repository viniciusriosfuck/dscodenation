#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from loguru import logger


# In[2]:


# Algumas configurações para o matplotlib.
from IPython.core.pylabtools import figsize
# %matplotlib inline


figsize(12, 8)

sns.set()


# In[3]:


fifa = pd.read_csv("fifa.csv")


# In[4]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
                   ]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.


# * This dataset was imported from Kaggle and corresponds to Fifa19 players
# * For the purpose of the AceleraDev Codenation DataScience Week 6 Challenge some columns were dropped
# * Such process result in a dataset named `fifa` with 18,207 rows and 37 columns
# * All of the kept columns contain numerical data
#  * there are 3 columns `["Age", "Overall", "Potential"]` that contain 18,207 `int64` entries each and
#  * each of the 34 left ones contain 18,159 non-null `float64` and 48 nan
# 

# In[6]:


fifa.shape


# In[7]:


fifa.head()


# In[8]:


fifa.info()


# In[9]:


fifa.isna().sum()


# In[10]:


fifa.columns


# In[11]:


#Criando um dataframe auxliar para analisar a consistencia das variaveis
cons = pd.DataFrame({'column' : fifa.columns,
                    'dtype': fifa.dtypes,
                    'missing' : fifa.isna().sum(),
                    'length' : fifa.shape[0],
                    'nunique': fifa.nunique()})
cons['missing_pct'] = round(cons['missing'] / cons['length'],2)
cons


# In[12]:


fifa.describe()


# * Due to the high amount of available data, the rows with `nan` data will be dropped
# * Notice that there are the same rows for all columns with `float64`
# * Now all columns contain numerical data
# 

# In[13]:


fifa.dropna(inplace=True)
fifa.describe()


# In[14]:


plt.figure(figsize = (20,20))
sns.heatmap(fifa.corr().round(1), annot= True);


# ## PCA

# In[15]:


pca = PCA(n_components=2)

projected = pca.fit_transform(fifa)

print(f"Original shape: {fifa.shape}, projected shape: {projected.shape}")
sns.scatterplot(projected[:, 0], projected[:, 1]);


# In[16]:


pca = PCA().fit(fifa)
evr = pca.explained_variance_ratio_
evr


# In[17]:


cumulative_variance_ratio = np.cumsum(evr)
cumulative_variance_ratio


# In[18]:


nr_PC_95pct = np.argmax(cumulative_variance_ratio >= 0.95) + 1
nr_PC_95pct


# In[19]:


g = sns.lineplot(np.arange(len(evr)), cumulative_variance_ratio)
g.axes.axhline(0.95, ls="--", color="red")
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');


# ### Attempt to rank the variables with PCA results

# In[20]:


fifa_pca = pca.transform(fifa)

# number of components
n_pcs= pca.components_.shape[0]

# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = fifa.columns
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

# LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
df = pd.DataFrame(dic.items())
df


# In[22]:


def create_importance_dataframe(pca, original_num_df):
    '''
    arguments: 
    original_num_df: the original numeric dataframe
    pca: the PCA model fitted object
    
    return: 
    importance_df: dataframe with PCA values assigned to the variables
    '''
    # Change pcs components ndarray to a dataframe
    importance_df = pd.DataFrame(pca.components_)

    # Assign columns
    importance_df.columns = original_num_df.columns

    # Change to absolute values
    importance_df = importance_df.apply(np.abs).T

    ## First get number of pcs
    num_pcs = importance_df.shape[1]

    ## Generate the new column names
    new_columns = [f'PC{i}' for i in range(1, num_pcs + 1)]

    ## Now rename
    importance_df.columns = new_columns

    # Return importance df
    return importance_df

# Call function to create importance df
importance_df = create_importance_dataframe(pca, fifa)

# Show first few rows
print(importance_df.head())

# Sort depending on PC of interest

## PC1 top 10 important features
pc1_top_10_features = importance_df['PC1'].sort_values(ascending = False)[:10]
print(), print(f'PC1 top 10 features are \n')
print(pc1_top_10_features)

## PC2 top 10 important features
pc2_top_10_features = importance_df['PC2'].sort_values(ascending = False)[:10]
print(), print(f'PC2 top 10 features are \n')
print(pc2_top_10_features)


# In[23]:


importance_df.mean(axis=1).sort_values(ascending = False)


# In[24]:


importance_PC = importance_df.mean()
importance_PC.sort_values(ascending = False)


# In[25]:


importance_df.rank(axis=0,ascending=False)


# In[26]:


importance_df.iloc[:,:nr_PC_95pct].rank(axis=0,ascending=False)


# * For the n PCs that are necessary to explain 95% variance, this could give a idea of important role of the variable.
# * The small values present most importance on the n selected PCs.
# 
# * Such approach has a lot of failures:
# 1. not consider the importance of each PC
# 2. not consider the importance with values of each variable
# 

# In[27]:


importance_df.iloc[:,:nr_PC_95pct].rank(axis=0,ascending=False).mean(axis=1).sort_values()


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[28]:


def q1():
    pca = PCA().fit(fifa)
    evr = pca.explained_variance_ratio_
    
    return evr[0].round(3)


# In[29]:


#Test
q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[30]:


def q2():
    pca = PCA().fit(fifa)
    evr = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(evr)
    nr_PC_95pct = np.argmax(cumulative_variance_ratio >= 0.95) + 1   # Python starts at 0
    
    return nr_PC_95pct


# In[31]:


#Test
q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[32]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
     ]


# In[33]:


x_projected = np.matmul(pca.components_, x)


# In[34]:


def q3():
    pca = PCA().fit(fifa)
    x_projected = np.matmul(pca.components_, x)
    return (x_projected[0].round(3), x_projected[1].round(3))


# In[35]:


# Test
q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[36]:


y_train = fifa['Overall']
X_train = fifa.drop(columns=['Overall'])


# In[37]:


reg = LinearRegression()
reg.fit(X_train, y_train)
rfe = RFE(reg)
rfe.fit(X_train,y_train)


# In[38]:


rfe_df = pd.DataFrame({'coluna':X_train.columns,
              'bool': rfe.get_support(),
              'coeficientes': pd.Series(reg.coef_)})
rfe_df


# In[39]:


rfe_df['coeficientes_abs'] = abs(rfe_df['coeficientes'])
rfe_df.nlargest(5, 'coeficientes_abs')['coluna'].tolist()


# In[40]:


rfe_df.nlargest(5, 'coeficientes_abs')['coluna'].tolist()


# In[41]:


def q4():
    
    y_train = fifa['Overall']
    X_train = fifa.drop(columns=['Overall'])
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    
    rfe = RFE(reg)
    rfe.fit(X_train,y_train)
    
    rfe_df = pd.DataFrame({'coluna':X_train.columns,
              'bool': rfe.get_support(),
              'coeficientes': pd.Series(reg.coef_)})
    rfe_df['coeficientes_abs'] = abs(rfe_df['coeficientes'])
    
    list_5 = rfe_df.nlargest(5, 'coeficientes_abs')['coluna'].tolist()
    
    return list_5


# In[42]:


# Test
q4()

