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


from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfVectorizer, TfidfTransformer
)
from sklearn.datasets import fetch_20newsgroups


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Countries DataSet

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


# ### Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# In[5]:


np.sort(countries["Region"].unique())


# In[6]:


countries.info()


# In[7]:


for col in new_column_names:
    try:
        countries[col] = countries[col].str.strip()  # trim leading spaces
        countries[col] = countries[col].str.replace(',','.').astype(float)
    except:
        pass


# A transformação pode ser constatada com

# In[8]:


countries.head()


# In[9]:


np.sort(countries["Region"].unique())


# In[10]:


countries.info()


# ### Análise

# O data_set conta com 20 variáveis de 227 países, sendo
# * 2 variáveis `object` (string)
# * 2 variáveis `int64`
# * 16 variáveis `float64`
# 

# As variáveis faltantes por categoria são:

# In[11]:


countries.isna().sum()


# As quantidades de valores únicos por categoria são:

# In[12]:


countries.nunique()


# A descrição estatística das variáveis numéricas é:

# In[13]:


countries.describe()


# O clima está rotulado em 6 categorias, sendo que alguns dados não se encontram em nenhuma delas.

# In[14]:


countries['Climate'].value_counts()


# A distribuição por região é a seguinte

# In[15]:


countries['Region'].value_counts()


# ### Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[16]:


def q1():
    return np.sort(countries["Region"].unique()).tolist()


# In[17]:


# 
q1()


# ### Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[18]:


def q2():
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    pop_density_10bins = est.fit_transform(countries[['Pop_density']])
    plt.hist(pop_density_10bins)
    above_90 = np.sum(pop_density_10bins >= 9)
    # (pop_density_10bins > np.percentile(pop_density_10bins, 90)).sum()
    return above_90


# In[19]:


q2()


# In[20]:


countries['Pop_density'].describe()


# Sem a discretização, o histograma não era visual.

# In[21]:


countries[['Pop_density']].hist();


# Por causa dos outliers superiores

# In[22]:


countries[['Pop_density']].boxplot();


# In[23]:


countries[['Pop_density']].plot(kind='box', logy=True);


# ### Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[24]:


def q3():
    
    enc = OneHotEncoder()
       
    climate_one_hot = enc.fit_transform(countries[['Climate']].astype('str'))
    # climate_one_hot = enc.fit_transform(countries[['Climate']].dropna())
    
    region_one_hot = enc.fit_transform(countries[['Region']])
    
    climate_old_n = countries['Climate'].nunique()
    region_old_n = countries['Region'].nunique()
    
    new_elements_climate = climate_one_hot.shape[1]
    new_elements_region = region_one_hot.shape[1]
    
    new_elements = new_elements_climate + new_elements_region
    
    return new_elements


# In[25]:


q3()


# #### Análises das etapas

# In[26]:


enc = OneHotEncoder()
    
climate_one_hot = enc.fit_transform(countries[['Climate']].dropna())
region_one_hot = enc.fit_transform(countries[['Region']])
        
climate_old_n = countries['Climate'].nunique()
region_old_n = countries['Region'].nunique()
    
new_elements_climate = climate_one_hot.shape[1] - 1
new_elements_region = region_one_hot.shape[1] - 1
    
new_elements = new_elements_climate + new_elements_region
new_elements_climate
new_elements_region


# In[27]:


climate_one_hot.shape[0] * climate_one_hot.shape[1] - len(countries['Climate'])


# In[28]:


new_elements = climate_one_hot.shape[0] * climate_one_hot.shape[1] - len(countries['Climate'])
new_elements = new_elements + region_one_hot.shape[0] * region_one_hot.shape[1] - len(countries['Region'])
new_elements


# In[29]:


climate_one_hot


# In[30]:


countries['Climate'].nunique()


# In[31]:


countries['Region'].nunique()


# In[32]:


region_one_hot 


# In[33]:


region_one_hot.toarray()[:10]


# In[34]:


enc.categories_[0]


# Em formato numérico não roda com NaN.

# In[35]:


enc = OneHotEncoder()
climate_one_hot = enc.fit_transform(countries[['Climate']].dropna())
enc.categories_[0]


# Se converte em string para contabilizar NaN.

# In[36]:


enc = OneHotEncoder()
climate_one_hot = enc.fit_transform(countries[['Climate']].astype('str'))
enc.categories_[0]


# In[37]:


print(countries['Climate'].unique())
len(countries['Climate'].unique())


# In[38]:


climate_one_hot.shape[1]


# In[39]:


print(countries['Region'].unique())
len(countries['Region'].unique())


# In[40]:


region_one_hot.shape[1]


# ### Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[41]:


# Single sample as list
test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[42]:


def q4():  #-1.047
    # Set the pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("standard_scaler", StandardScaler())
    ])
    
    # Get list with numeric columns names from a dataframe
    names_numeric_col = countries.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Fit and transform the numeric subset of a dataframe with the numeric pipeline
    countries_transformed = num_pipeline.fit_transform(countries[names_numeric_col])
    
    # Discard the string entries from a list
    test_country_num_list = [x for x in test_country if not isinstance(x, str)]  # not select strings
    
    # Convert a single sample inputted as a list to sklearn input np.array
    test_country_num_np = np.asarray(test_country_num_list).reshape(1, -1)
    
    # Return index from Arable in the numeric subset
    idx_Arable = names_numeric_col.index('Arable')
    
    # Apply the pipeline transformation
    test_country_transformed = num_pipeline.transform(test_country_num_np)
    
    # Get transformed Arable entry in the required format
    arable_transformed = round(test_country_transformed[:,idx_Arable][0],3)
    return arable_transformed


# In[43]:


q4()


# Mostrando as etapas:

# In[44]:


# Set the pipeline
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("standard_scaler", StandardScaler())
])


# In[45]:


# Get list with numeric columns names from a dataframe
names_numeric_col = countries.select_dtypes(include=['int64', 'float64']).columns.tolist()


# In[46]:


# Fit and transform the numeric subset of a dataframe with the numeric pipeline
countries_transformed = num_pipeline.fit_transform(countries[names_numeric_col])


# In[47]:


# Discard the string entries from a list
test_country_num_list = [x for x in test_country if not isinstance(x, str)]  # not select strings
test_country_num_list


# In[48]:


# Convert a single sample inputted as a list to sklearn input np.array
test_country_num_np = np.asarray(test_country_num_list).reshape(1, -1)
test_country_num_np


# In[49]:


# Return index from Arable in the numeric subset
idx_Arable = names_numeric_col.index('Arable')
idx_Arable


# In[50]:


# Apply the pipeline transformation
test_country_transformed = num_pipeline.transform(test_country_num_np)
test_country_transformed


# In[51]:


# Get transformed Arable entry in the required format
arable_transformed = round(test_country_transformed[:,idx_Arable][0],3)
arable_transformed


# #### Análise da padronização

# Para verificar o processo de normalização, poderia ser inspecionadas as variáveis média e desvio padrão, para ver se correspondem a nulas e unitárias para todas as variáveis numéricas.

# In[52]:


countries.head()


# In[53]:


# data_transformed = countries.select_dtypes(exclude=['int64', 'float64']).copy()
df_countries_transformed = pd.DataFrame(countries_transformed, columns=names_numeric_col)


# In[54]:


df_countries_transformed.head()


# In[55]:


df_countries_transformed.mean()


# In[56]:


df_countries_transformed.std()


# O processo de padronização permite comparar as features visualmente em um boxplot:

# In[57]:


df_countries_transformed.boxplot(rot=90);


# #### Comentário sobre o dado test_country
# 
# Faria mais sentido o input test_country se referir a um dado já transformado.
# Por exemplo, o input fornecido conta com uma população negativa.
# 
# Como observado no plot abaixo, ele apresenta valores razoáveis para todas as variáveis numéricas.

# In[58]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.barh(names_numeric_col,test_country_num_list)
plt.gca().invert_yaxis()


# Caso fosse esse o caso, para aplicar a transformação inversa, o pipeline poderia ser reduzido à etapa de padronização.
# 
# Pois a imputação de NaN com mediana, não é inversível no sklearn.

# In[59]:


# Set the pipeline
num_pipeline2 = Pipeline(steps=[
    ("standard_scaler", StandardScaler())
])


# In[60]:


# Fit and transform the numeric subset of a dataframe with the numeric pipeline
countries_transformed_2 = num_pipeline2.fit_transform(countries[names_numeric_col])


# E os dados se refeririam a um país com esses valores numéricos que, de fato, parecem razoáveis.

# In[61]:


test_country_inv_transformed = num_pipeline2.inverse_transform(test_country_num_np)
dftest_country_inv_transformed = pd.DataFrame(test_country_inv_transformed, columns=names_numeric_col)
dftest_country_inv_transformed


# #### Formas alternativas

# In[62]:


col_int_float_names = countries.select_dtypes(include=['int64', 'float64']).columns

imputed_countries = countries.copy()

for col in col_int_float_names:
    imputed_countries[col] = countries[col].fillna(countries[col].median())


# In[63]:


countries.head()


# In[64]:


imputed_countries.head()


# In[65]:


# Create the Scaler object
scaler = StandardScaler()

scaled_countries = imputed_countries.copy()

for col in col_int_float_names:
    scaled_countries[[col]] = scaler.fit_transform(scaled_countries[[col]])
    


# In[66]:


scaled_countries.head()


# ### Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[67]:


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


# In[68]:


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


# In[69]:


q5()


# #### Análises

# In[70]:


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
outliers.hist(bins=25);


# quartile_boxplot e np.quantile apresentam diferenças ...

# In[71]:


x = scaled_countries['Net_migration'].to_numpy()
Q_1, Q_3 = quartile_boxplot(x)
Q_1


# In[72]:


np.quantile(x, 0.25)


# In[73]:


Q_3


# In[74]:


np.quantile(x, 0.75)


# In[75]:


scaled_countries['Net_migration'].describe()


# ## Newsgroups dataset - Text Analysis
# 
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 

# In[76]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# Quantas vezes as palavras `the` e `phone` aparecem no subdataset utilizado?

# In[77]:


count_vectorizer = CountVectorizer()
newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
words_idx = sorted([count_vectorizer.vocabulary_.get(f"{word.lower()}") for word in
                    [u"the", u"phone"]])

count_the_phone = pd.DataFrame(newsgroups_counts[:, words_idx].toarray(),
                               columns=np.array(count_vectorizer.get_feature_names())[words_idx])
count_the_phone.sum()


# Quais as taxas de frequências dos termos `the` e `phone` no subdataset utilizado?

# In[78]:


tfidf_transformer = TfidfTransformer()

tfidf_transformer.fit(newsgroups_counts)

newsgroups_tfidf = tfidf_transformer.transform(newsgroups_counts)
tfidf_the_phone = pd.DataFrame(newsgroups_tfidf[:, words_idx].toarray(),
                               columns=np.array(count_vectorizer.get_feature_names())[words_idx])
tfidf_the_phone.sum()


# ### Questão 6
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[79]:


def q6():
        
    cv = CountVectorizer(vocabulary=['phone'])
    phone_count = cv.fit_transform(newsgroups.data).toarray().sum()
    
    return phone_count


# In[80]:


q6()


# ### Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[81]:


def q7():
    
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
    
    words_idx = sorted([count_vectorizer.vocabulary_.get(f"{word.lower()}")
                        for word in [u"the", u"phone"]])

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(newsgroups.data)
    
    
    newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroups.data)
    phone_tfidf = pd.DataFrame(newsgroups_tfidf_vectorized[:, words_idx].toarray(),
                               columns=np.array(count_vectorizer.get_feature_names())[words_idx]).sum().loc['phone'].round(3)
    return phone_tfidf


# In[82]:


q7()


# In[83]:


def q7_1():  # NOT WORKING
        
    pipe = Pipeline([('count', TfidfVectorizer(vocabulary=['phone'])),
                  ('tfid', TfidfTransformer())]).fit(newsgroup['data'])
    pipe['count'].transform(newsgroup['data']).toarray()
    X = pipe['tfid'].idf_
       
    
    return round(X[0], 3)


# In[ ]:




