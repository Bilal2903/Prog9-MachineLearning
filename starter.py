"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9
Deze code is geschreven in Python3
Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- scikit-learn
"""
from machinelearningdata import Machine_Learning_Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

def extract_from_json_as_np_array(key, json_data):
    """ helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn """
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)

STUDENTNUMMER = "1032518"

assert STUDENTNUMMER != "1234567", "Verander 1234567 in je eigen studentnummer"

print("STARTER CODE")

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(STUDENTNUMMER)


# UNSUPERVISED LEARNING

# haal clustering data op
kmeans_training = data.clustering_training()

# extract de x waarden
X = extract_from_json_as_np_array("x", kmeans_training)

#print(X)

# slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
x = X[...,0]
y = X[...,1]

# teken de punten
for i in range(len(x)):
    plt.plot(x[i], y[i], 'k.') # k = zwart

plt.axis([min(x), max(x), min(y), max(y)])
plt.show()

# TODO: print deze punten uit en omcirkel de mogelijke clusters
# kijk ../Images/clusters.png
# Ik denk dat er 3 clusters zijn

# TODO: ontdek de clusters mbv kmeans en teken een plot met kleurtjes
km = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X)

color_theme = np.array(['red', 'purple', 'blue', 'pink'])

plt.subplot(1,2,1)
plt.title("K-means")
centroids = np.array(km.cluster_centers_)
plt.scatter(x=x, y=y, c=color_theme[km.labels_])
plt.scatter(centroids[:,0], centroids[:,1], marker="x", color="green")
plt.show()

# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)

# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = extract_from_json_as_np_array("y", classification_training)

# teken de punten
for i in range(len(X)):
    if Y[i] == 0:
        plt.plot(X[...,0][i], X[...,1][i], 'r.')
    else:
        plt.plot(X[..., 0][i], X[..., 1][i], 'b.')

plt.title("Classification")
plt.show()

# TODO: leer de classificaties
lore = LogisticRegression()
lore.fit(X, Y)

detr = tree.DecisionTreeClassifier()
detr = detr.fit(X, Y)

# TODO: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#       bijvoordeeld Y_predict

Y_pred = lore.predict(X)
tree_pred = detr.predict(X)

# TODO: vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt
acc_class = accuracy_score(Y, Y_pred)
acc_tree = accuracy_score(Y, tree_pred)

print("Accuracy decision tree: {:.2f}".format(acc_tree)) #0.77
print("Accuracy logistical regression: {:.2f}".format(acc_class)) #0.72

tree.plot_tree(detr)

# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)

# TODO: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
#       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
#       geen echte Y-waarden gekregen hebt.
#       onderstaande code stuurt je voorspelling naar de server, die je dan
#       vertelt hoeveel er goed waren.

Z = detr.predict(X_test)
ZZ = lore.predict(X_test)

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie dec tree (test): " + str(classification_test)) #0.79

classification_test = data.classification_test(ZZ.tolist())
print("Classificatie accuratie log reg (test): " + str(classification_test)) # 0.78

# Op basis van deze dat (zie terminal) zou ik kiezen voor decision tree, want de accuratie is hoger in beide gevallen.