import pytest

from numeros import NotANumber, numero_a_palabras, palabras_a_numero, precio_en_palabras


@pytest.mark.parametrize(
    "n, words",
    [
        (0, "cero"), (1, "uno"), (16, "dieciséis"), (21, "veintiuno"), (22, "veintidós"), (30, "treinta"),
        (31, "treinta y uno"), (99, "noventa y nueve"), (100, "cien"), (101, "ciento uno"),
        (115, "ciento quince"), (200, "doscientos"), (555, "quinientos cincuenta y cinco"),
        (777, "setecientos setenta y siete"), (999, "novecientos noventa y nueve"), (1000, "mil"),
        (1001, "mil uno"), (2100, "dos mil cien"), (21000, "veintiún mil"),
    ],
)  # fmt: skip
def test_numero_a_palabras(n, words):
    assert numero_a_palabras(n) == words


@pytest.mark.parametrize(
    "n, words",
    [(1, "un"), (21, "veintiún"), (31, "treinta y un"), (101, "ciento un"), (11, "once"), (111, "ciento once")],
)
def test_uno_shortens_before_a_noun(n, words):
    assert numero_a_palabras(n, before_noun=True) == words


@pytest.mark.parametrize(
    "price, words",
    [
        (1, "un dólar"), (21, "veintiún dólares"), (140.30, "ciento cuarenta dólares con treinta centavos"),
        (64.01, "sesenta y cuatro dólares con un centavo"), (999.49, "novecientos noventa y nueve dólares con cuarenta y nueve centavos"),
    ],
)  # fmt: skip
def test_precio_en_palabras(price, words):
    assert precio_en_palabras(price) == words


@pytest.mark.parametrize(
    "text, value, in_words",
    [
        ("120", 120, False), ("$120.50", 120.5, False), ("120,5", 120.5, False), ("ciento veinte", 120, True),
        ("Ciento veinte dólares con cincuenta", 120.5, True), ("veintitres", 23, True), ("veintiún dólares", 21, True),
        ("cien", 100, True), ("mil doscientos", 1200, True), ("cuesta setenta y cinco dolares", 75, True),
        ("sietecientos", 700, True),
    ],
)  # fmt: skip
def test_palabras_a_numero(text, value, in_words):
    assert palabras_a_numero(text) == (pytest.approx(value), in_words)


def test_every_written_number_reads_back():
    for n in range(1000):
        assert palabras_a_numero(numero_a_palabras(n))[0] == n
        assert palabras_a_numero(precio_en_palabras(n + 0.25))[0] == pytest.approx(n + 0.25)


@pytest.mark.parametrize("text", ["", "mucho", "ciento perro"])
def test_unreadable_guesses(text):
    with pytest.raises(NotANumber):
        palabras_a_numero(text)
