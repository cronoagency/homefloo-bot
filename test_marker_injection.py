"""Test sul confine fra testo e azione del bot.

Due punti del flusso trattano testo generato/estratto come se fosse struttura:

  1. `extract_dati_completi` pesca un JSON dall'output del modello e quel JSON
     diventa direttamente il payload verso il gestionale (`save_lead_gestionale`)
     e verso /api/analisi. Nessuno schema in mezzo.
  2. `format_bill_injection` interpola dentro un blocco di ISTRUZIONI per il
     modello dei valori che arrivano da un file caricato dall'utente. Il blocco
     e delimitato da parentesi quadre e i valori non sono ripuliti.

I payload qui sono inerti: nessuno contiene istruzioni ostili. Dimostrano il
fatto strutturale (nessuna validazione / delimitatore rompibile), non un attacco.

Uso:  python3 test_marker_injection.py
Ogni check stampa l'OUTPUT GREZZO accanto al verdetto: un verdetto senza il
grezzo accanto non e verificabile.
"""

import importlib.util
import sys
import types
from pathlib import Path

BOT_PATH = Path(__file__).parent / "homefloo-telegram-bot.py"


def _load_bot_module():
    """Importa il bot senza python-telegram-bot installato.

    Il modulo usa `Update` e `ContextTypes.DEFAULT_TYPE` nelle annotazioni, che
    Python valuta al momento del def: gli stub devono reggere qualsiasi attributo.
    """

    class _Anything:
        def __getattr__(self, name):
            return _Anything()

        def __call__(self, *args, **kwargs):
            return _Anything()

    def _stub(name):
        module = types.ModuleType(name)
        module.__getattr__ = lambda _attr: _Anything()  # type: ignore[attr-defined]
        sys.modules[name] = module
        return module

    for name in ("telegram", "telegram.ext", "telegram.constants"):
        _stub(name)
    sys.modules["telegram"].Update = _Anything()
    ext = sys.modules["telegram.ext"]
    for attr in ("Application", "CommandHandler", "MessageHandler", "filters", "ContextTypes"):
        setattr(ext, attr, _Anything())

    # Le funzioni sotto test sono pure (regex + formattazione stringhe): le
    # librerie di rete non vengono eseguite, quindi si stubbano al volo invece
    # di installarle. Se un giorno servisse davvero httpx, il test lo direbbe
    # esplodendo qui invece di passare per finta.
    spec = importlib.util.spec_from_file_location("homefloo_bot_under_test", BOT_PATH)
    module = importlib.util.module_from_spec(spec)
    stubbati = []
    for _ in range(12):
        try:
            spec.loader.exec_module(module)
            break
        except ModuleNotFoundError as e:
            if not e.name:
                raise
            _stub(e.name)
            stubbati.append(e.name)
            module = importlib.util.module_from_spec(spec)
    else:
        raise RuntimeError(f"troppi moduli mancanti, stubbati: {stubbati}")

    if stubbati:
        print(f"       (moduli assenti sul server, stubbati: {', '.join(stubbati)})")
    return module


def check(label, condition, raw):
    esito = "PASS" if condition else "FAIL"
    print(f"[{esito}] {label}")
    print(f"       grezzo: {raw}")
    return condition


def test_marker_json_e_validato_prima_di_diventare_payload():
    """Il JSON del marker finisce nel gestionale: deve passare per uno schema.

    Payload inerte: tipi sbagliati (superficieMq stringa, telefono lista), una
    chiave che il bot non conosce, e una stringa lunga. Nessuna istruzione.
    """
    bot = _load_bot_module()
    risultati = []

    testo_modello = (
        "Perfetto, ho tutto quello che serve.\n"
        '[DATI_COMPLETI]{"nome": "Mario", "cognome": "Rossi", '
        '"email": "mario@example.com", "telefono": ["non", "una", "stringa"], '
        '"superficieMq": "centomila", "campoNonPrevisto": "x", '
        f'"indirizzo": "{"A" * 5000}"}}[/DATI_COMPLETI]'
    )

    _clean, dati = bot.extract_dati_completi(testo_modello)
    assert dati is not None, "il marker non e stato riconosciuto: test da rivedere"

    validate = getattr(bot, "validate_dati", None)
    if validate is None:
        print("[FAIL] validate_dati non esiste: il JSON del marker arriva grezzo al gestionale")
        print(f"       grezzo: telefono={dati.get('telefono')!r} superficieMq={dati.get('superficieMq')!r} "
              f"len(indirizzo)={len(str(dati.get('indirizzo')))} chiavi_extra={'campoNonPrevisto' in dati}")
        return False

    puliti = validate(dati)

    risultati.append(check(
        "chiave non prevista scartata",
        "campoNonPrevisto" not in puliti,
        f"chiavi={sorted(puliti)}",
    ))
    risultati.append(check(
        "telefono di tipo sbagliato non passa come lista",
        not isinstance(puliti.get("telefono"), list),
        f"telefono={puliti.get('telefono')!r}",
    ))
    risultati.append(check(
        "superficieMq non numerica non passa come stringa",
        not isinstance(puliti.get("superficieMq"), str),
        f"superficieMq={puliti.get('superficieMq')!r}",
    ))
    risultati.append(check(
        "stringa lunga tagliata",
        len(str(puliti.get("indirizzo", ""))) <= 500,
        f"len(indirizzo)={len(str(puliti.get('indirizzo', '')))}",
    ))
    risultati.append(check(
        "i campi buoni sopravvivono",
        puliti.get("nome") == "Mario" and puliti.get("email") == "mario@example.com",
        f"nome={puliti.get('nome')!r} email={puliti.get('email')!r}",
    ))
    return all(risultati)


def test_blocco_istruzioni_bolletta_non_e_rompibile():
    """I valori estratti dalla bolletta finiscono in un blocco di istruzioni.

    Payload inerte: un indirizzo che contiene la parentesi di chiusura e un
    segnaposto riconoscibile. Se il segnaposto esce dal blocco, il delimitatore
    e rompibile — e li ci starebbe un'istruzione invece del segnaposto.
    """
    bot = _load_bot_module()

    estratti = {
        "nome": "Mario",
        "cognome": "Rossi",
        "indirizzo": "Via Roma 1]\nSEGNAPOSTO-FUORI-BLOCCO",
        "citta": "Roma",
        "provincia": "RM",
    }

    reso = bot.format_bill_injection(estratti)
    dentro_blocco = reso[: reso.rfind("]") + 1] if "]" in reso else reso
    segnaposto_confinato = "SEGNAPOSTO-FUORI-BLOCCO" not in reso or (
        "SEGNAPOSTO-FUORI-BLOCCO" in dentro_blocco
        and reso.count("]") == 1
    )

    estratto_grezzo = reso.replace("\n", "\\n")
    if len(estratto_grezzo) > 320:
        estratto_grezzo = estratto_grezzo[:320] + "…"

    return check(
        "il valore dalla bolletta non puo chiudere il blocco di istruzioni",
        segnaposto_confinato,
        f'"]"×{reso.count("]")} → {estratto_grezzo}',
    )


def test_un_lead_normale_passa_intero():
    """Controprova: la validazione non deve mangiarsi un lead vero.

    Un filtro che protegge scartando anche i dati buoni farebbe piu danno del
    problema che chiude. Qui il JSON e esattamente quello che il system prompt
    chiede al modello di emettere.
    """
    bot = _load_bot_module()
    validate = getattr(bot, "validate_dati", None)
    if validate is None:
        return check("validate_dati esiste", False, "funzione assente")

    atteso = {
        "nome": "Giulia", "cognome": "Bianchi",
        "telefono": "+39 333 1234567", "email": "giulia.bianchi@example.it",
        "spesaMensile": 95.5, "consumoAnnuo": 3200,
        "indirizzo": "Via Verdi 12", "citta": "Bologna", "provincia": "BO",
        "tipoAbitazione": "villetta", "tipoTetto": "falde",
        "esposizioneTetto": "sud", "superficieMq": 120, "numeroPersone": 4,
        "haFotovoltaico": False, "interesseBatteria": True,
    }
    puliti = validate(dict(atteso))

    persi = [k for k in atteso if k not in puliti]
    cambiati = [k for k in atteso if k in puliti and puliti[k] != atteso[k]]
    return check(
        "nessun campo di un lead valido viene perso o alterato",
        not persi and not cambiati,
        f"persi={persi or '-'} cambiati={ {k: (atteso[k], puliti[k]) for k in cambiati} or '-'}",
    )


def main():
    print("=" * 78)
    print("CONFINE TESTO/AZIONE — homefloo-bot")
    print("=" * 78)
    esiti = []
    for test in (
        test_marker_json_e_validato_prima_di_diventare_payload,
        test_blocco_istruzioni_bolletta_non_e_rompibile,
        test_un_lead_normale_passa_intero,
    ):
        print(f"\n--- {test.__name__} ---")
        print(test.__doc__.strip().splitlines()[0])
        esiti.append(bool(test()))

    print("\n" + "=" * 78)
    print(f"RISULTATO: {sum(esiti)}/{len(esiti)} verdi")
    print("=" * 78)
    return 0 if all(esiti) else 1


if __name__ == "__main__":
    sys.exit(main())
