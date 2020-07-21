#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn import preprocessing


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
 
for col in new_column_names:
    try:
        countries[col] = countries[col].str.strip()  # trim leading spaces
        countries[col] = countries[col].str.replace(',','.').astype(float)
    except:
        pass


# In[6]:


countries.info()


# In[7]:


countries.head()


# In[8]:


np.sort(countries["Region"].unique())


# In[ ]:





# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[9]:


def q1():
    return np.sort(countries["Region"].unique()).tolist()


# In[10]:


# 
q1()


# In[ ]:





# In[11]:


countries['Pop_density'].describe()


# In[12]:



est = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
pop_density_10bins = est.fit_transform(countries[['Pop_density']])
plt.hist(pop_density_10bins)
(pop_density_10bins > np.percentile(pop_density_10bins, 90)).sum()


# In[13]:


enc = preprocessing.OneHotEncoder()
    
climate_one_hot = enc.fit_transform(countries[['Climate']].dropna())
region_one_hot = enc.fit_transform(countries[['Region']])
        
climate_old_n = countries['Climate'].nunique()
region_old_n = countries['Region'].nunique()
    
new_elements_climate = climate_one_hot.shape[1] - 1
new_elements_region = region_one_hot.shape[1] - 1
    
new_elements = new_elements_climate + new_elements_region
new_elements_climate
new_elements_region


# In[14]:


climate_one_hot.shape[0] * climate_one_hot.shape[1] - len(countries['Climate'])


# In[15]:


new_elements = climate_one_hot.shape[0] * climate_one_hot.shape[1] - len(countries['Climate'])
new_elements = new_elements + region_one_hot.shape[0] * region_one_hot.shape[1] - len(countries['Region'])
new_elements


# In[16]:


climate_one_hot


# In[17]:


countries['Climate'].nunique()


# In[18]:


countries['Region'].nunique()


# In[19]:


region_one_hot 


# In[20]:


countries['Climate'].value_counts()


# In[21]:


countries['Region'].value_counts()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[22]:


def q2():
    est = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    pop_density_10bins = est.fit_transform(countries[['Pop_density']])
    plt.hist(pop_density_10bins)
    above_90 = np.sum(pop_density_10bins >= 9)
    # (pop_density_10bins > np.percentile(pop_density_10bins, 90)).sum()
    return above_90


# In[23]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[24]:


def q3():
    enc = preprocessing.OneHotEncoder()
    
    climate_one_hot = enc.fit_transform(countries[['Climate']].dropna())
    region_one_hot = enc.fit_transform(countries[['Region']])
        
    climate_old_n = countries['Climate'].nunique()
    region_old_n = countries['Region'].nunique()
    
    new_elements_climate = climate_one_hot.shape[1] - 1
    new_elements_region = region_one_hot.shape[1] - 1
    
    new_elements = new_elements_climate + new_elements_region
    
    return new_elements


# In[25]:


q3()  #18


# In[26]:


climate_one_hot.shape[1]


# In[27]:


climate_one_hot.shape[1]


# In[28]:


col_int_float_names = countries.select_dtypes(include=['int64', 'float64']).columns

imputed_countries = countries.copy()

for col in col_int_float_names:
    imputed_countries[col] = countries[col].fillna(countries[col].median())


# In[29]:


countries.head()


# In[30]:


imputed_countries.head()


# In[ ]:





# In[31]:


# Create the Scaler object
scaler = preprocessing.StandardScaler()

scaled_countries = imputed_countries.copy()

for col in col_int_float_names:
    scaled_countries[[col]] = scaler.fit_transform(scaled_countries[[col]])
    


# In[32]:


scaled_countries.head()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[33]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[34]:


def q4():  #-1.047
    # Retorne aqui o resultado da questão 4.
    pass


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[35]:


statistics = scaled_countries['Net_migration'].describe()
Q_1 = statistics['25%']
Q_3 = statistics['75%']
IQR = Q_3 - Q_1

outliers = scaled_countries['Net_migration'].loc[(scaled_countries['Net_migration'] < Q_1 - 1.5 * IQR) 
                                          | (scaled_countries['Net_migration'] > Q_3 + 1.5 * IQR)]

outliers_below = scaled_countries['Net_migration'].loc[(scaled_countries['Net_migration'] < Q_1 - 1.5 * IQR)]

outliers_above = scaled_countries['Net_migration'].loc[(scaled_countries['Net_migration'] > Q_3 + 1.5 * IQR)]

scaled_countries[['Net_migration']].boxplot()
fig = plt.figure()
outliers.hist(bins=25)


# quartile_boxplot e np.quantile apresentam diferenças ...

# In[36]:


def quartile_boxplot(x):
    
    X = np.sort(x)
    
    n = len(X)
    median = np.median(X)
    
    if n % 2 != 0:  # odd
        # Median excluded
        X_below = X[:n//2]
        X_above = X[n//2+1:]
        
    else:  # even
        # Median included
        X_below = X[:n//2]
        X_above = X[n//2:]
        
    Q_1 = np.median(X_below)
    
    
    # TODO: investigate differences
    # assert(np.allclose(np.quantile(X, 0.25), Q_1))
    
    Q_3 = np.median(X_above)
    
    # assert(np.allclose(np.quantile(X, 0.75), Q_3))
    
    return Q_1, Q_3


# In[37]:


x = scaled_countries['Net_migration'].to_numpy()
Q_1, Q_3 = quartile_boxplot(x)
Q_1


# In[38]:


np.quantile(x, 0.25)


# In[39]:


Q_3


# In[40]:


np.quantile(x, 0.75)


# In[41]:


scaled_countries['Net_migration'].describe()


# In[42]:


def q5():
    # statistics = scaled_countries['Net_migration'].describe()
    #Q_1 = statistics['25%']
    # Q_3 = statistics['75%']
    
    # Q_1 = np.quantile(scaled_countries['Net_migration'].to_numpy(), 0.25)
    # Q_3 = np.quantile(scaled_countries['Net_migration'].to_numpy(), 0.75)
    
    Q_1, Q_3 = quartile_boxplot(scaled_countries['Net_migration'].to_numpy())
    
    IQR = Q_3 - Q_1
    
    outliers_below = scaled_countries['Net_migration'].loc[(scaled_countries['Net_migration'] < Q_1 - 1.5 * IQR)]

    outliers_above = scaled_countries['Net_migration'].loc[(scaled_countries['Net_migration'] > Q_3 + 1.5 * IQR)]

    remove = False

    return (len(outliers_below), len(outliers_above), remove)


# In[43]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[44]:


from sklearn.datasets import fetch_20newsgroups

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[45]:


def q6():
    from sklearn.feature_extraction.text import CountVectorizer
    
    cv = CountVectorizer(vocabulary=['phone'])
    phone_count = cv.fit_transform(newsgroup['data']).toarray().sum()
    
    return phone_count


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[50]:


def q7():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    
    cv = CountVectorizer(vocabulary=['phone'])
    phone_count_vec = cv.fit_transform(newsgroup['data'])  # .toarray()
    
    tfidf_transformer=TfidfTransformer()  # smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(phone_count_vec)
    
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
    
    return round(df_idf.loc['phone'].item(), 3)


# In[51]:


q7()


# In[ ]:




