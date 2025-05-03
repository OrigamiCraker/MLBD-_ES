# Lavorare con dati testuali
# In questo esercizio lavorerai con dati testuali. I dati testuali sono solitamente rappresentati come stringhe,
# composte da caratteri di lunghezza variabile. Questa caratteristica è molto diversa da quella delle feature numeriche,
# per cui sarà necessario pre-processare i dati prima di poter applicare algoritmi di machine learning.

# Applicare il modello Bag-of-Words a un dataset di esempio
# Per costruire un modello bag-of-words basato sul conteggio delle parole nei rispettivi documenti,
# possiamo usare la classe CountVectorizer implementata in scikit-learn.
# Come vedremo nella sezione di codice seguente, la classe CountVectorizer prende in input un array di dati testuali,
# che possono essere documenti o semplici frasi, e costruisce automaticamente il modello bag-of-words.

# Dati testuali e costruzione del vocabolario:

import numpy as np

docs = np.array ([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining the weather is sweet and one and one is two'
])

from sklearn.feature_extraction.text import  CountVectorizer # CountVectorizer è una classe che permette la creazione delle bag of words tramite la tokenizzaione, accessibile trmite l'attributo "vocabulary_"
count = CountVectorizer()
count.fit(docs)

# .fit():
    # Analizza tutti i documenti.
    # Costruisce un dizionario interno (vocabolario) di tutte le parole uniche presenti.
    # Ogni parola viene associata a un indice.
    # Non trasforma ancora i dati in numeri (non crea la matrice bag-of-words), fa solo la fase di "apprendimento" del vocabolario.

print('Dimensione del vocabolario: {}'.format(len(count.vocabulary_)))
print('Contenuto del vocabolario:\n {}'.format(count.vocabulary_))

# Per la creazione della rappresentazione della BOW del training dataset possiamo chiamare il metodo transform():
bag = count.transform((docs))
# Repr ritorna una stringa contenente una rappresenzaione dell'oggetto dato in input di tipo stampabile. Rappresentazione sparsa della BOW (tanti 0 e 1)
print('BOW: {}'.format(repr(bag)))
print('Rappresentazione densa della BOW:\n {}'.format(bag.toarray()))

#testo
