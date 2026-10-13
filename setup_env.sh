#!/bin/bash
# Création de l'environnement virtuel
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo 'Environnement prêt.'
