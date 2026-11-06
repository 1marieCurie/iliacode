# ============================================================
#  IA Générative, séance 1 --- la brique durable, le moteur adopté
#  Quatre fonctions à écrire (parties 2 à 4 du TP G1), chacune se termine
#  par une ligne raise NotImplementedError à remplacer par le vrai code.
#  Les séances suivantes relisent ces fonctions telles quelles, pour
#  générer (NLP S7), affiner (NLP S9), étendre le vocabulaire (NLP S10), servir et
#  alléger le moteur (IA Générative S2 et S3).
#  La fonction surprise_par_caractere, en bas du fichier, est déjà écrite.
# ============================================================
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MOTEUR_PAR_DEFAUT = "Qwen/Qwen2.5-Coder-7B"


def charger_moteur(nom=MOTEUR_PAR_DEFAUT):
    """Charge le moteur ouvert et son tokeniseur, figés, sur le GPU s'il
    y en a un, sinon sur le processeur. Rend le couple (modele, tokeniseur).
    nom est le nom du moteur sur Hugging Face ou le chemin d'un dossier
    local qui contient ses fichiers."""
    raise NotImplementedError("partie 2 du TP, à compléter")


def compter_parametres(modele):
    """Le nombre N de paramètres du moteur, la somme, sur toutes ses
    matrices et tous ses vecteurs, du produit de leurs dimensions."""
    raise NotImplementedError("partie 3 du TP, à compléter")


def memoire_poids(modele):
    """La mémoire des poids en octets, chaque paramètre occupant autant
    d'octets que son format, 2 en FP16, 4 en FP32."""
    raise NotImplementedError("partie 3 du TP, à compléter")


def probas_moteur(modele, tokeniseur, texte):
    """La liste des probabilités que le moteur a données à chaque token
    réellement écrit de texte, sachant ceux qui le précèdent. Le premier
    token n'a pas d'histoire, il n'est pas noté. Cette liste est l'entrée
    des fonctions entropie_croisee et perplexite de la séance 3."""
    raise NotImplementedError("partie 4 du TP, à compléter")


# ------------------------------------------------------------
#  Déjà écrite. La surprise par caractère compare deux moteurs qui ne
#  découpent pas le texte de la même façon, la surprise totale du texte,
#  en bits, divisée par son nombre de caractères.
# ------------------------------------------------------------
def surprise_par_caractere(probas, texte):
    """La surprise totale des tokens notés, en bits, divisée par le nombre
    de caractères du texte."""
    if not probas or not texte:
        return math.inf
    total = sum(math.inf if p == 0 else -math.log2(p) for p in probas)
    return total / len(texte)
