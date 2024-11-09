Rapport de Projet : Détection de Plaques d'Immatriculation avec Mask R-CNN
1. Introduction
Dans le cadre de ce projet, nous avons utilisé un modèle Mask R-CNN pour détecter les plaques d'immatriculation de voitures. Ce modèle est basé sur le modèle pré-entraîné COCO, auquel nous avons adapté un entraînement pour notre jeu de données spécifique de plaques d'immatriculation.

2. Environnement et Préparation des Données

a. Préparation du Dataset
Un dataset de 300 images a été collecté à partir de différentes sources d'images de plaques d'immatriculation. Ces images ont été annotées à l'aide de l'outil LabelMe pour marquer les plaques d'immatriculation.

Répartition des images :

Entraînement : 225 images
Validation : 50 images
Production : 25 images

b. Hyperparamètres du Modèle pour l'Entraînement

Les hyperparamètres utilisés pour l'entraînement du modèle sont les suivants :

BACKBONE : resnet101
BACKBONE_STRIDES : [4, 8, 16, 32, 64]
BATCH_SIZE : 2
BBOX_STD_DEV : [0.1, 0.1, 0.2, 0.2]
COMPUTE_BACKBONE_SHAPE : None
DETECTION_MAX_INSTANCES : 100
DETECTION_MIN_CONFIDENCE : 0.9
DETECTION_NMS_THRESHOLD : 0.3
FPN_CLASSIF_FC_LAYERS_SIZE : 1024
GPU_COUNT : 2
GRADIENT_CLIP_NORM : 5.0
IMAGES_PER_GPU : 1
IMAGE_CHANNEL_COUNT : 3
IMAGE_MAX_DIM : 1024
IMAGE_META_SIZE : 14
IMAGE_MIN_DIM : 800
IMAGE_MIN_SCALE : 0
IMAGE_RESIZE_MODE : square
IMAGE_SHAPE : [1024, 1024, 3]
LEARNING_MOMENTUM : 0.9
LEARNING_RATE : 0.001
LOSS_WEIGHTS : {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 1.0}
MASK_POOL_SIZE : 14
MASK_SHAPE : [28, 28]
MAX_GT_INSTANCES : 100
MEAN_PIXEL : [123.7, 116.8, 103.9]
MINI_MASK_SHAPE : (56, 56)
NAME : custom
NUM_CLASSES : 2
POOL_SIZE : 7
POST_NMS_ROIS_INFERENCE : 1000
POST_NMS_ROIS_TRAINING : 2000
PRE_NMS_LIMIT : 6000
ROI_POSITIVE_RATIO : 0.33
RPN_ANCHOR_RATIOS : [0.5, 1, 2]
RPN_ANCHOR_SCALES : (32, 64, 128, 256, 512)
RPN_ANCHOR_STRIDE : 1
RPN_BBOX_STD_DEV : [0.1, 0.1, 0.2, 0.2]
RPN_NMS_THRESHOLD : 0.7
RPN_TRAIN_ANCHORS_PER_IMAGE : 256
STEPS_PER_EPOCH : 100
TOP_DOWN_PYRAMID_SIZE : 256
TRAIN_BN : False
TRAIN_ROIS_PER_IMAGE : 200
USE_MINI_MASK : True
USE_RPN_ROIS : True
VALIDATION_STEPS : 50
WEIGHT_DECAY : 0.0001


3. Résultats de l'Entraînement

a. Fichier Log d'Entraînement
Le fichier de log d'entraînement suit les itérations et les différentes pertes (loss) à chaque époque. La dernière ligne du fichier log montre les pertes finales après l'entraînement.

Extrait de log (dernière ligne) :

arduino
Copy code
Epoch 30/30
100/100 [==============================] - 120s 1s/step - loss: 0.0221 - rpn_class_loss: 0.0011 - rpn_bbox_loss: 0.0104 - mrcnn_class_loss: 0.0032 - mrcnn_bbox_loss: 0.0039 - mrcnn_mask_loss: 0.0035 - val_loss: 0.0197 - val_rpn_class_loss: 0.0009 - val_rpn_bbox_loss: 0.0098 - val_mrcnn_class_loss: 0.0029 - val_mrcnn_bbox_loss: 0.0035 - val_mrcnn_mask_loss: 0.0036

Pour des logs plus détaillés, voir le fichier logs.txt.

b. Matrice de Confusion
La matrice de confusion suivante montre les performances du modèle sur le jeu de test, mettant en évidence le nombre de faux positifs et de faux négatifs ainsi que les objets correctement identifiés.


c. Graphique des Pertes d'Entraînement et de Validation
Le graphique ci-dessous illustre l'évolution des pertes d'entraînement et de validation pendant les 20 époques.

![alt text](image-1.png)

d. Commentaires sur les Résultats du Modèle
Le modèle montre une bonne convergence des pertes, et les métriques de validation indiquent que le modèle est capable de détecter correctement la plupart des plaques d'immatriculation dans le jeu de validation. Cependant, du fine tuning pourrait être nécessaire pour réduire les faux positifs et les faux négatifs restants.

5. Fine-tuning
Pour le fine-tuning du modèle, les paramètres suivants ont été utilisés pour améliorer les performances :

