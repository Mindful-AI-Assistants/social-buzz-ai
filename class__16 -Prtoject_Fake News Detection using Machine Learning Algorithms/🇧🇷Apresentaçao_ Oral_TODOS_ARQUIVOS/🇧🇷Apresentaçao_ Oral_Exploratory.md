

<br>

## Apresentação Oral

Bom dia a todos!
Hoje vou apresentar o resultado de um estudo acadêmico voltado para detecção de Fake News, utilizando técnicas e algoritmos de *Machine Learning*.

<br>

### 1. Introdução

Fake News são informações falsas, geralmente divulgadas pelas redes sociais, que causam impactos graves. Como circular notícias falsas é fácil, precisamos de métodos para ajudar a identificar o que é verdade ou não.

<br>

### 2. Objetivo

O objetivo deste trabalho foi analisar diferentes algoritmos para detectar Fake News automaticamente, buscando qual modelo é mais eficiente.

<br>

### 3. Metodologia

Coletamos muitas notícias (falsas e verdadeiras) do Kaggle.
Limpamos, padronizamos e embaralhamos os dados. Analisamos visualmente com WordCloud e separamos em treino e teste.

<br>

### 4. Algoritmos Testados

Usamos: Regressão Logística, Decision Tree, Random Forest, SVM, KNN.
Cada modelo "aprende" a identificar notícias falsas.

<br>

### 5. Resultados

Quatro modelos tiveram acurácia maior que 90%. Decision Tree foi o melhor.
KNN teve mais erros.
Analisamos acurácia, precisão, sensibilidade, especificidade.

<br>

### 6. Limitações e Futuro

Dificuldade em bases de dados em português.
Deve-se testar mais algoritmos, ampliar dados e validar melhor.


<br>

### 7. Conclusão

ML é poderoso para detectar Fake News e proteger a sociedade!


<br>

### Esse código instala as bibliotecas que vamos usar. Elas são como caixas de ferramentas para programar.

<br>

This code installs the libraries we will use. They are like toolboxes for programming.

```python
!pip install wordcloud seaborn nltk
```

<br>

### Importamos as ferramentas que acabamos de instalar. Assim, podemos usar seus superpoderes no código.

We import the tools we just installed. This way, we can use their superpowers in the code.

<br>

```python
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
```


<br>

### Aqui carregamos as notícias falsas e verdadeiras que vamos estudar. Imagine dois livros diferentes!

Here we load the fake and real news we will study. Imagine two different books!

<br>

```python
fake = pd.read_csv('Fake.csv')   # Suba 'Fake.csv' ao Colab / Upload 'Fake.csv' to Colab
true = pd.read_csv('True.csv')   # Suba 'True.csv' ao Colab / Upload 'True.csv' to Colab
fake['target'] = 1  # Fake News
true['target'] = 0  # True News
```


<br>

### Juntamos tudo em uma grande tabela e embaralhamos, como cartas de baralho! Removemos título e data para focar só no texto.

We put everything together in a big table and shuffle it, like deck of cards! We remove title and date to focus only on the text.

```python
data = pd.concat([fake, true], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)
data.drop(['title', 'date'], axis=1, inplace=True)
```


<br>

### Vamos deixar as palavras mais simples: só minúsculas, sem pontuação nem palavras que não ajudam, como “a, de, o”.

Let’s make words simpler: lowercase only, without punctuation or words that don’t help, like “a, the, of”.

<br>

```python
stop_words = set(stopwords.words('portuguese'))
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

data['text'] = data['text'].apply(clean_text)
```

<br>

### Veja um desenho bonito com as palavras que mais aparecem nas notícias, a nuvem de palavras.

See a pretty drawing with the words that show up most in the news, the word cloud.

<br>

```python
wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['text']))
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
```


<br>

### Transformamos os textos em números que os computadores entendem, e separamos em dados para ensinar (“treino”) e testar (“teste”) o computador.

We turn texts into numbers that computers understand and split them in “training” and “testing” data for the computer to learn and practice.

```python
X = data['text']
y = data['target']
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
```


<br>

### Agora vem a magia! Vamos treinar cinco tipos de máquinas inteligentes para descobrir se uma notícia é falsa ou verdadeira.

Now come the magic! We will train five smart machines to tell if news is fake or real.

<br>

```python
modelos = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}
for nome, modelo in modelos.items():
    print(f'\n--- {nome} ---')
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {nome}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
```


<br>

### Se quiser guardar uma máquina treinada para usar depois, basta salvar!

If you want to save a trained machine to use later, just save it!

<br>

```python
# from joblib import dump, load
# dump(modelos['Decision Tree'], 'decision_tree_model.joblib')
```


<br>

**Conclusão**
Machine Learning se mostrou promissor para detectar Fake News. Com pesquisa, podemos construir um mundo mais seguro para todos!

