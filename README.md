# ChronoFace - Module de Vieillissement (Partie 2)

**Auteur :** Ouriel Mimoun  
**Projet :** ChronoFace - Analyse et Synthèse Temporelle de Visages  
**Technologie :** CycleGAN (PyTorch)

---

## 📋 Description
Ce module implémente la fonctionnalité de **synthèse de vieillissement et de rajeunissement** du projet ChronoFace. Il utilise une architecture **CycleGAN** (Generative Adversarial Network) pour effectuer un transfert de style "Image-to-Image" sur des données non appariées (Unpaired).

Contrairement aux approches classiques, ce modèle a été **entraîné "from scratch"** (depuis zéro) sur le dataset UTKFace, apprenant à dissocier la structure du visage (identité) de l'attribut temporel (âge).

### Fonctionnalités Clés :
* **Vieillissement :** Transformation d'un visage "Jeune" vers "Vieux".
* **Rajeunissement :** Transformation d'un visage "Vieux" vers "Jeune".
* **Préservation d'identité :** Utilisation de la *Cycle Consistency Loss* pour garantir que la personne reste reconnaissable.

---

## 🛠️ Installation

### Prérequis
* Python 3.8+
* PyTorch (avec support CUDA recommandé)
* Bibliothèques listées dans `requirements.txt`

### Installation des dépendances
```bash
pip install -r requirements.txt
