# ============================================================
# Séance 3 — Tests du premier moteur et de sa mesure
# Lancer : pytest -v
# Partie 1 : pytest -v -k p1
# ============================================================

import math
import re
import pytest

from bigramme import (
    DEBUT, FIN_SEQ, encadrer, compter, proba, predire,
    proba_lissee, probas_du_texte
)
from perplexite import surprise, entropie_croisee, perplexite
from hygiene import masquer_secrets, MASQUE


# Mini-corpus de la séance : cinq lignes découpées en mots
MINI = [
    ["for", "ticket", "in", "tickets"],
    ["for", "ticket", "in", "queue"],
    ["for", "user", "in", "users"],
    ["if", "ticket", "is", "None"],
    ["for", "ticket", "in", "queue"],
]

# 10 mots distincts + le token de fin
TAILLE_V_MINI = 11


def tables():
    """Reconstruit les tables de comptage du mini-corpus."""
    return compter(MINI)


def produit(probabilites):
    resultat = 1.0
    for p in probabilites:
        resultat *= p
    return resultat


# --- Partie 1 : compter --------------------------------------

def test_p1_encadrer():
    assert encadrer(["if", "ticket"]) == [
        DEBUT, "if", "ticket", FIN_SEQ
    ]


def test_p1_compter_la_table():
    paires, contextes = tables()

    assert contextes[DEBUT] == 5
    assert contextes["in"] == 4
    assert contextes["ticket"] == 4
    assert contextes["tickets"] == 1

    assert paires[("in", "tickets")] == 1
    assert paires[("in", "queue")] == 2
    assert paires[("in", "users")] == 1

    assert paires[("ticket", "queue")] ==  0
    assert paires[("ticket", "in")] == 3

    assert contextes[FIN_SEQ] == 0


# --- Partie 2 : prédire --------------------------------------

def test_p2_proba_lue_dans_la_table():
    paires, contextes = tables()

    assert proba("for", "ticket", paires, contextes) == pytest.approx(3 / 4)
    assert proba(DEBUT, "for", paires, contextes) == pytest.approx(4 / 5)
    assert proba("ticket", "queue", paires, contextes) == 0


def test_p2_predire_apres_in():
    assert predire("in", *tables()) == [
        ("queue", 0.5),
        ("tickets", 0.25),
        ("users", 0.25),
    ]


def test_p2_proba_d_une_ligne():
    paires, contextes = tables()

    vue = probas_du_texte(
        [["for", "ticket", "in", "tickets"]],
        paires, contextes, TAILLE_V_MINI, 0
    )

    nouvelle = probas_du_texte(
        [["for", "user", "in", "queue"]],
        paires, contextes, TAILLE_V_MINI, 0
    )

    cassee = probas_du_texte(
        [["if", "user", "is", "None"]],
        paires, contextes, TAILLE_V_MINI, 0
    )

    # Les transitions présentes dans le corpus
    assert proba("for", "user", paires, contextes) == pytest.approx(1 / 4)
    assert proba("in", "queue", paires, contextes) == pytest.approx(2 / 4)

    # La première ligne est entièrement observée
    assert produit(vue) > 0

    # La nouvelle ligne contient la transition "user" -> "in",
    # qui existe, mais "in" -> "queue" existe également.
    # Le produit doit donc être positif.
    assert produit(nouvelle) > 0

    # La transition "if" -> "user" n'existe pas dans le corpus.
    assert proba("if", "user", paires, contextes) == 0
    assert produit(cassee) == 0



# --- Partie 3 : lisser ---------------------------------------

def test_p3_lissage_supprime_le_zero():
    paires, contextes = tables()

    assert proba_lissee(
        "if", "user", paires, contextes, TAILLE_V_MINI, 1
    ) == pytest.approx(1 / 12)

    lissee = probas_du_texte(
        [["if", "user", "is", "None"]],
        paires, contextes, TAILLE_V_MINI, 1
    )

    assert all(p > 0 for p in lissee)
    assert produit(lissee) > 0


def test_p3_lissage_reste_une_probabilite():
    paires, contextes = tables()

    vocabulaire = {mot for ligne in MINI for mot in ligne}
    vocabulaire.add(FIN_SEQ)

    assert len(vocabulaire) == TAILLE_V_MINI

    for k in (1, 0.1):
        total = sum(
            proba_lissee(
                "in", mot, paires, contextes, TAILLE_V_MINI, k
            )
            for mot in vocabulaire
        )
        assert total == pytest.approx(1)


# --- Partie 4 : mesurer --------------------------------------

def test_p4_surprise():
    assert surprise(1) == 0
    assert surprise(0.5) == 1
    assert surprise(0.25) == 2
    assert surprise(0.125) == 3
    assert surprise(0) == math.inf


def test_p4_entropie_et_perplexite_de_l_exemple():
    probas = [0.5, 0.25, 0.5, 0.125]

    assert entropie_croisee(probas) == pytest.approx(1.75)
    assert perplexite(probas) == pytest.approx(2 ** 1.75)


def test_p4_perplexite_d_un_modele_uniforme():
    assert perplexite([1 / 8] * 20) == pytest.approx(8)
    assert perplexite([1.0] * 20) == pytest.approx(1)


def test_p4_un_zero_rend_la_perplexite_infinie():
    paires, contextes = tables()

    cassee = probas_du_texte(
        [["if", "user", "is", "None"]],
        paires, contextes, TAILLE_V_MINI, 0
    )

    assert perplexite(cassee) == math.inf


# --- Partie 5 : mesurer un modèle ----------------------------

def test_p5_perplexite_du_bigramme_lisse():
    paires, contextes = tables()

    vue = probas_du_texte(
        [["for", "ticket", "in", "tickets"]],
        paires, contextes, TAILLE_V_MINI, 1
    )

    nouvelle = probas_du_texte(
        [["if", "user", "is", "None"]],
        paires, contextes, TAILLE_V_MINI, 1
    )

    assert all(p > 0 for p in vue)
    assert all(p > 0 for p in nouvelle)

    # La perplexité est calculée à partir des probabilités
    # réellement produites par le modèle.
    assert perplexite(vue) >= 1
    assert perplexite(nouvelle) >= 1


# --- Partie 6 : protéger -------------------------------------

LIGNE_SECRETE = 'API_KEY = "sk-4f9a2c71"'


def decouper(ligne):
    """Découpe une ligne en mots, nombres et ponctuation."""
    return re.findall(r"\w+|[^\w\s]", ligne)


def test_p6_masquer_secrets():
    assert masquer_secrets(LIGNE_SECRETE) == (
        'API_KEY = "' + MASQUE + '"'
    )

    assert masquer_secrets("mot_de_passe: 'azerty123'") == (
        "mot_de_passe: '" + MASQUE + "'"
    )

    assert masquer_secrets('titre = "Ticket ouvert"') == (
        'titre = "Ticket ouvert"'
    )


def test_p6_le_secret_ne_doit_plus_etre_present():
    depot = [
        "for ticket in tickets",
        LIGNE_SECRETE,
        "if ticket is None",
    ]

    lignes_propres = [
        masquer_secrets(ligne) for ligne in depot
    ]

    texte_propre = "\n".join(lignes_propres)

    assert "4f9a2c71" not in texte_propre
    assert MASQUE in texte_propre

    # Vérifie que le découpage des lignes nettoyées
    # ne contient plus la valeur secrète.
    tokens_propres = [
        decouper(ligne) for ligne in lignes_propres
    ]

    assert all(
        "4f9a2c71" not in token
        for ligne in tokens_propres
        for token in ligne
    )

    paires, _ = compter(tokens_propres)

    assert all(
        "4f9a2c71" not in mot
        for paire in paires
        for mot in paire
    )
