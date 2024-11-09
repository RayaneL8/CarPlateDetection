import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Supposons que tu as les données suivantes :
# - y_true : Les labels réels des images du jeu de test
# - y_pred : Les prédictions du modèle sur ces images (par exemple, sous forme de classes)
# - history : L'objet d'historique retourné par l'entraînement du modèle (ce qui contient les pertes d'entraînement et de validation)

# Exemple de valeurs pour y_true et y_pred (à remplacer par tes propres données)
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]  # Labels réels
y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 0, 0]  # Prédictions du modèle

# Calcul de la matrice de confusion
cm = confusion_matrix(y_true, y_pred)

# Création de la matrice de confusion sous forme de graphique
def plot_confusion_matrix(cm, class_names, title='Matrice de Confusion'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Prédictions')
    plt.ylabel('Véritables')
    plt.show()

# Noms des classes (à adapter à ton dataset)
class_names = ['Plaque non détectée', 'Plaque détectée']
plot_confusion_matrix(cm, class_names)

# ============================
# Graphique des Pertes d'Entraînement et Validation
# ============================

# Exemple de données d'historique (remplacer avec l'historique de ton entraînement)
history = {
    'loss': [1.8921, 0.9924,0.8347, 0.7134,0.6478, 0.5243,0.4589, 0.4456 , 0.4017,   0.3950 ,  0.3620,  0.3548,  0.3016, 0.3056,  0.2944, 0.2776,0.2707 ,0.2296, 0.2349, 0.2353],  # Pertes d'entraînement
    'val_loss': [1.1738, 1.0651,0.9295 , 0.9442,0.8252,  0.7633,  0.7886,  0.7301, 0.6952,0.6590,0.6598,  0.6395 , 0.6272,0.5797 , 0.6034 , 0.6189,0.6063,0.6764 , 0.5952, 0.6115 ],  # Pertes de validation
    'epochs': [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  # Numéro des époques
}

# Création du graphique des pertes d'entraînement et de validation
def plot_loss_graph(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history['epochs'], history['loss'], label='Perte d\'Entraînement', color='blue')
    plt.plot(history['epochs'], history['val_loss'], label='Perte de Validation', color='red')
    plt.title('Pertes d\'Entraînement et de Validation')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss_graph(history)
