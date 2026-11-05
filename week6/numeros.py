"""Spanish numbers for the price game: writing prices out in words, and reading a learner's guess back.

Both directions are deterministic on purpose - the game grades the learner's Spanish, so the answer key must not
come from an LLM that might itself misspell "setecientos" or forget that "veintiún dólares" drops its final -o.
"""

import re
import unicodedata

UNITS = [
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve",
    "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete",
    "veintiocho", "veintinueve",
]  # fmt: skip
TENS = {30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta", 70: "setenta", 80: "ochenta", 90: "noventa"}
HUNDREDS = {
    100: "ciento", 200: "doscientos", 300: "trescientos", 400: "cuatrocientos", 500: "quinientos",
    600: "seiscientos", 700: "setecientos", 800: "ochocientos", 900: "novecientos",
}  # fmt: skip


def _below_thousand(n):
    if n == 100:
        return "cien"
    parts = []
    if n >= 100:
        parts.append(HUNDREDS[n // 100 * 100])
        n %= 100
    if n >= 30:
        parts.append(TENS[n // 10 * 10] + (f" y {UNITS[n % 10]}" if n % 10 else ""))
    elif n or not parts:
        parts.append(UNITS[n])
    return " ".join(parts)


def numero_a_palabras(n, before_noun=False):
    """0 <= n < 1,000,000 in words. With `before_noun`, "uno" shortens as it does before a masculine noun:
    "un dólar", "veintiún dólares", "treinta y un dólares"."""
    if not 0 <= n < 1_000_000:
        raise ValueError(f"{n} is out of range")
    thousands, rest = divmod(n, 1000)
    words = []
    if thousands:
        # "mil", never "un mil"; and "veintiún mil" because "mil" is itself a masculine noun
        words.append("mil" if thousands == 1 else f"{_shorten_uno(_below_thousand(thousands))} mil")
    if rest or not thousands:
        words.append(_below_thousand(rest))
    text = " ".join(words)
    return _shorten_uno(text) if before_noun and rest % 10 == 1 and rest % 100 != 11 else text


def _shorten_uno(text):
    if text.endswith("veintiuno"):
        return text[: -len("veintiuno")] + "veintiún"
    if text.endswith("uno"):
        return text[:-1]
    return text


def precio_en_palabras(price):
    """$140.30 -> "ciento cuarenta dólares con treinta centavos"."""
    dollars, cents = divmod(round(price * 100), 100)
    text = f"{numero_a_palabras(dollars, before_noun=True)} {'dólar' if dollars == 1 else 'dólares'}"
    if cents:
        text += f" con {numero_a_palabras(cents, before_noun=True)} {'centavo' if cents == 1 else 'centavos'}"
    return text


# ---------------------------------------------------------------------------------------------------------------
# Reading a guess


def _plain(word):
    """Lower case without accents, so "veintitres" and "dieciseis" are accepted as well."""
    return "".join(c for c in unicodedata.normalize("NFD", word.lower()) if unicodedata.category(c) != "Mn")


VALUES = {_plain(word): value for value, word in enumerate(UNITS)}
VALUES.update({"un": 1, "una": 1, "veintiun": 21, "veintiuna": 21, "cien": 100})
VALUES.update({_plain(word): value for value, word in TENS.items()})
VALUES.update({_plain(word): value for value, word in HUNDREDS.items()})
# Learner misspellings that are still unambiguous: read them, and let the game show the right spelling
VALUES.update({"sietecientos": 700, "nuevecientos": 900, "cincocientos": 500, "cientos": 100})

IGNORED = {"y", "dolar", "dolares", "pesos", "euros", "con", "centavo", "centavos", "de", "cuesta", "son", "es"}


class NotANumber(ValueError):
    """The guess contains a word that is not part of a Spanish number."""


def _words_to_int(words):
    total, current = 0, 0
    for word in words:
        if word == "mil":
            total += (current or 1) * 1000
            current = 0
        elif word in VALUES:
            current += VALUES[word]
        else:
            raise NotANumber(word)
    return total + current


def palabras_a_numero(text):
    """A learner's guess -> (dollars as a float, whether it was written in words).

    Accepts "ciento veinte", "ciento veinte dólares con cincuenta", "$120.50", "120,50" and "120".
    Raises NotANumber naming the first word it could not read.
    """
    text = text.strip()
    digits = re.fullmatch(r"\$?\s*(\d+)(?:[.,](\d{1,2}))?\s*\$?", text)
    if digits:
        cents = int((digits.group(2) or "0").ljust(2, "0"))
        return int(digits.group(1)) + cents / 100, False

    words = [_plain(w) for w in re.findall(r"[^\W\d_]+", text)]
    if not words:
        raise NotANumber(text)
    # "con" separates dollars from cents: "veinte dólares con cincuenta (centavos)"
    if "con" in words:
        split = words.index("con")
        dollar_words, cent_words = words[:split], words[split + 1 :]
    else:
        dollar_words, cent_words = words, []
    keep = lambda ws: [w for w in ws if w not in IGNORED]  # noqa: E731
    dollars = _words_to_int(keep(dollar_words)) if keep(dollar_words) else 0
    cents = _words_to_int(keep(cent_words)) if keep(cent_words) else 0
    if not keep(dollar_words) and not keep(cent_words):
        raise NotANumber(text)
    return dollars + cents / 100, True
