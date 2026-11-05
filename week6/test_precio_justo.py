from precio_justo import judge
from pricer.items import Item

ITEM = Item(title="Kettle", category="Appliances", price=140.3, summary="Title: Kettle")


def test_digits_are_said_back_in_words():
    feedback, won, cards = judge("120", ITEM, model_guess=200)
    assert "se dice: *ciento veinte dólares*" in feedback
    assert "ciento cuarenta dólares con treinta centavos" in feedback
    assert won  # 20.30 off beats 59.70 off
    assert cards == [{"spanish": "ciento cuarenta dólares (140 $)", "english": "140 dollars"}]


def test_correct_spelling_is_praised():
    feedback, _, _ = judge("ciento cuarenta dólares", ITEM, model_guess=140)
    assert "¡Bien escrito!" in feedback


def test_misspelling_is_corrected():
    feedback, won, _ = judge("sietecientos", ITEM, model_guess=150)
    assert "se escribe: *setecientos dólares*" in feedback
    assert not won


def test_cents_in_words():
    feedback, _, _ = judge("ciento cuarenta con treinta", ITEM, model_guess=0)
    assert "¡Bien escrito!" in feedback
