"""
Homefloo Telegram Bot — Claude Haiku per conversazione, Claude Sonnet per analisi.

Claude Haiku fa la conversazione e raccolta dati.
Quando i dati sono completi, chiama /api/analisi per il report (Claude Sonnet).
Supporta foto e PDF bolletta — estrae dati anagrafici + li tiene per l'analisi.

Config via env vars:
    HOMEFLOO_TELEGRAM_TOKEN (required)
    GESTIONALE_API_KEY (required)
    ANTHROPIC_API_KEY (required — per conversazione e estrazione bolletta)
    HOMEFLOO_API (optional)
    GESTIONALE_API (optional)
    DATA_DIR (optional, default /app/data)
"""

import base64
import logging
import json
import os
import re
import time
from pathlib import Path
from io import BytesIO

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# --- Config from env vars ---
TELEGRAM_TOKEN = os.environ.get("HOMEFLOO_TELEGRAM_TOKEN", "")
GESTIONALE_API_KEY = os.environ.get("GESTIONALE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HOMEFLOO_API = os.environ.get(
    "HOMEFLOO_API",
    "http://gcgoo88g4ow8sccokc0o48c4.91.98.89.69.sslip.io/api/analisi",
)
GESTIONALE_API = os.environ.get(
    "GESTIONALE_API",
    "https://gestionale.homefloo.com/api/analisi-energetiche/bot",
)
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_HEADERS = {
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}
CHAT_MODEL = "claude-haiku-4-5-20251001"

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
SESSIONS_PATH = DATA_DIR / "sessions.json"
CONV_LOG_PATH = DATA_DIR / "conversations.jsonl"

MAX_HISTORY = 20  # max messaggi per sessione (10 turni user+assistant)
SESSION_EXPIRE_HOURS = 48  # sessioni piu vecchie di 48h vengono scartate
CHAT_TIMEOUT = 30  # secondi (Claude Haiku e veloce)
ANALISI_TIMEOUT = 180  # secondi (Claude Sonnet per analisi)

# Marker per dati completi
DATI_PATTERN = re.compile(r"\[DATI_COMPLETI\]\s*(.*?)\s*\[/DATI_COMPLETI\]", re.DOTALL)

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("homefloo-bot")

# --- Sessioni in memoria (con persistenza su disco) ---
# chat_id -> {"messages": [...], "last_active": timestamp, "file": {...} | None, "request_id": str | None}
sessions: dict[int, dict] = {}


def save_sessions():
    """Salva sessioni su disco (senza file bytes — troppo grandi)."""
    try:
        SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for cid, sess in sessions.items():
            serializable[str(cid)] = {
                "messages": sess.get("messages", []),
                "last_active": sess.get("last_active", 0),
                "request_id": sess.get("request_id"),
                "has_file": sess.get("file") is not None,
            }
        with open(SESSIONS_PATH, "w") as f:
            json.dump(serializable, f, ensure_ascii=False)
    except Exception as e:
        log.error(f"Errore salvataggio sessioni: {e}")


def load_sessions():
    """Carica sessioni da disco al boot. Scarta quelle scadute."""
    global sessions
    if not SESSIONS_PATH.exists():
        return
    try:
        with open(SESSIONS_PATH) as f:
            data = json.load(f)
        now = time.time()
        expire = SESSION_EXPIRE_HOURS * 3600
        for cid_str, sess in data.items():
            if now - sess.get("last_active", 0) < expire:
                sessions[int(cid_str)] = {
                    "messages": sess.get("messages", []),
                    "last_active": sess.get("last_active", 0),
                    "file": None,  # file bytes non persistiti
                    "request_id": sess.get("request_id"),
                }
        log.info(f"Caricate {len(sessions)} sessioni da disco")
    except Exception as e:
        log.error(f"Errore caricamento sessioni: {e}")


def log_conversation(chat_id: int, role: str, content: str, extra: dict | None = None):
    """Logga ogni messaggio su file JSONL per storico conversazioni."""
    CONV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chat_id": chat_id,
        "role": role,
        "content": content[:500],
    }
    if extra:
        entry.update(extra)
    with open(CONV_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_session(chat_id: int) -> dict:
    """Recupera o crea sessione per un utente."""
    if chat_id not in sessions:
        sessions[chat_id] = {"messages": [], "last_active": time.time(), "file": None, "request_id": None}
    sessions[chat_id]["last_active"] = time.time()
    return sessions[chat_id]


def trim_session(messages: list[dict]) -> list[dict]:
    """Taglia la sessione se troppo lunga."""
    if len(messages) > MAX_HISTORY:
        return messages[-MAX_HISTORY:]
    return messages


def extract_dati_completi(text: str) -> tuple[str, dict | None]:
    """Estrae i dati strutturati dal marker e ritorna (testo_pulito, dati_o_None)."""
    match = DATI_PATTERN.search(text)
    if not match:
        return text, None

    clean_text = text[:match.start()].rstrip()

    try:
        dati = json.loads(match.group(1).strip())
        return clean_text, dati
    except json.JSONDecodeError as e:
        log.warning(f"JSON non valido nel marker DATI_COMPLETI: {e}")
        return clean_text, None


HOMEFLOO_SYSTEM_PROMPT = """Sei l'assistente virtuale di Homefloo, azienda italiana specializzata in impianti fotovoltaici residenziali.

OBIETTIVO: guidare il cliente verso un'analisi energetica gratuita raccogliendo i dati necessari attraverso una conversazione naturale.

FLUSSO CONVERSAZIONE:
1. Saluta e chiedi come puoi aiutare
2. Chiedi se ha una bolletta da condividere (foto o dati)
3. Se il cliente ha caricato la bolletta:
   a. I dati anagrafici (nome, cognome, indirizzo) vengono estratti automaticamente dalla bolletta e ti verranno comunicati nel messaggio.
   b. Chiedi al cliente: "Dalla bolletta risulta che l'intestatario è [Nome Cognome] con indirizzo in [Via, Città]. Sono i dati corretti da usare per l'analisi?"
   c. Se confermati → usali e NON richiederli. Passa a raccogliere gli altri dati mancanti.
   d. Se il cliente dice che sono diversi → chiedi i dati corretti.
   e. NON chiedere spesa mensile né consumo — verranno estratti dalla bolletta.
4. Se il cliente comunica dati della bolletta a voce/testo: conferma quello che hai capito.
5. Se il cliente NON ha la bolletta E non l'ha caricata: chiedi almeno quanto spende al mese di luce (anche una stima va bene). Questo dato è ESSENZIALE — senza sapere il consumo non puoi dimensionare l'impianto. Insisti gentilmente.
6. Raccogli SOLO i dati che NON hai già. Se dalla bolletta hai nome, cognome e indirizzo, salta quei campi. Prima di chiedere, controlla nel contesto della conversazione se il dato è già stato fornito. I dati da raccogliere (se mancanti):
   - Spesa mensile in bolletta O consumo annuo in kWh (OBBLIGATORIO solo se NON ha caricato la bolletta)
   - Indirizzo dell'immobile (con città e provincia)
   - Tipo di abitazione (villetta, appartamento, casa indipendente)
   - Tipo di tetto (piano, a falde, non lo so)
   - Esposizione del tetto (sud, est, ovest, nord)
   - Superficie dell'abitazione in m²
   - Numero di persone in casa
   - Se ha già un impianto fotovoltaico
   - Se è interessato alla batteria di accumulo
7. Per i dati di contatto: se hai già nome e cognome dalla bolletta e il cliente li ha confermati, chiedi SOLO telefono ed email. Se non hai nome/cognome, chiedi anche quelli.
   Chiedi in modo naturale, ad esempio: "Per preparare il report personalizzato e poterti ricontattare, mi servirebbero un numero di telefono e un indirizzo email."
8. SOLO quando hai TUTTI i dati (immobile + contatto), conferma il riepilogo al cliente e dì che stai preparando l'analisi dettagliata. Alla FINE del messaggio (dopo il testo visibile al cliente), aggiungi il blocco JSON:
[DATI_COMPLETI]
{"nome":"string","cognome":"string","telefono":"string","email":"string","spesaMensile":number_or_null,"consumoAnnuo":number_or_null,"indirizzo":"string","citta":"string","provincia":"string","tipoAbitazione":"villetta|appartamento|casa_indipendente","tipoTetto":"piano|falde|non_so","esposizioneTetto":"sud|est|ovest|nord","superficieMq":number,"numeroPersone":number,"haFotovoltaico":false,"interesseBatteria":true}
[/DATI_COMPLETI]
ATTENZIONE: NON aggiungere MAI il blocco [DATI_COMPLETI] se mancano nome, cognome, telefono o email. Chiedi prima i dati mancanti. Se il cliente ha caricato la bolletta, metti spesaMensile e consumoAnnuo a null.
9. Dopo l'analisi, presenta i risultati in modo chiaro e proponi un appuntamento

GESTIONE UTENTE NEGATIVO:
- Se il cliente rifiuta di dare un dato o dice "no", NON insistere sullo stesso dato.
- Valuta cosa hai già (dalla bolletta, dalla conversazione precedente) e cosa manca davvero.
- Spiega in modo propositivo PERCHÉ serve quel dato specifico. Esempio: "Senza l'indirizzo non posso calcolare l'irraggiamento solare della tua zona, che è fondamentale per la stima."
- Se il cliente rifiuta dati ESSENZIALI (indirizzo O spesa/consumo quando non ha bolletta): spiega chiaramente che senza quei dati non puoi procedere con l'analisi, ma che un consulente può aiutarlo di persona. Esempio: "Capisco, nessun problema. Purtroppo senza almeno la città non riesco a fare la stima. Se vuoi, posso farti richiamare da un nostro consulente che può aiutarti di persona — ti serve solo lasciare un recapito."
- NON ripetere domande già fatte. Se hai già chiesto indirizzo e tipo abitazione, non rifarle.
- I dati ESSENZIALI senza i quali non puoi procedere sono: indirizzo (o almeno città+provincia) E (bolletta OPPURE spesa mensile OPPURE consumo annuo).
- Gli altri dati (tipo tetto, esposizione, superficie, persone) puoi stimarli con valori standard se il cliente non vuole rispondere.

TONO:
- Professionale ma accessibile, come un consulente di fiducia
- Mai commerciale o aggressivo
- Conservativo nelle stime
- Rispondi SEMPRE in italiano corretto — attenzione agli articoli (UN indirizzo, UNA email, IL numero) e alle concordanze di genere
- Risposte brevi e dirette — è una chat, non un documento

REGOLE:
- Non dare mai tempistiche di installazione
- Non menzionare marche o modelli specifici di pannelli/inverter
- Non garantire tempi di rientro esatti — sempre "stimato" o "indicativo"
- Se non hai un dato, chiedi — non inventare mai
- Se il cliente chiede qualcosa che non sai, dì che un consulente lo contatterà
- Se il cliente va fuori tema, rispondi brevemente e riporta la conversazione sul fotovoltaico in modo naturale
- Massimo 2-3 domande per messaggio
- Usa emoji con moderazione

DATI UTILI PER RISPOSTE RAPIDE:
- Consumo medio famiglia italiana: 2.700 kWh/anno
- Costo medio energia: ~0.25-0.30 euro/kWh
- Detrazione fiscale: 50% in 10 anni (prima casa, 2026)
- Tempo rientro tipico: 4-6 anni
- Pannello standard: 490 Wp, minimo 8 pannelli
- Batterie disponibili: 5, 7, 10, 14, 15, 21 kWh
- Nord Italia: ~1.100 kWh/kWp/anno
- Centro Italia: ~1.300 kWh/kWp/anno
- Sud Italia e isole: ~1.500 kWh/kWp/anno"""


async def call_claude(messages: list[dict]) -> str:
    """Chiama Claude Haiku per la conversazione."""
    # Filtra solo user/assistant messages
    chat_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]

    payload = {
        "model": CHAT_MODEL,
        "max_tokens": 1024,
        "system": HOMEFLOO_SYSTEM_PROMPT,
        "messages": chat_messages,
    }

    async with httpx.AsyncClient(timeout=CHAT_TIMEOUT) as client:
        response = await client.post(
            ANTHROPIC_API_URL,
            json=payload,
            headers={**ANTHROPIC_HEADERS, "x-api-key": ANTHROPIC_API_KEY},
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("content", [])
        if content and content[0].get("type") == "text":
            return content[0]["text"]
        return "Mi dispiace, non sono riuscito a generare una risposta."


async def find_existing_lead(chat_id: int) -> dict | None:
    """Cerca un lead esistente non archiviato per questo chat_id."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                GESTIONALE_API,
                params={"chatId": str(chat_id)},
                headers={"x-api-key": GESTIONALE_API_KEY},
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("found"):
                    log.info(f"[{chat_id}] Lead esistente trovato: {data.get('requestId')} (stato: {data.get('stato')})")
                    return data
            return None
    except Exception as e:
        log.error(f"[{chat_id}] Errore ricerca lead: {e}")
        return None


async def save_lead_gestionale(dati: dict, chat_id: int, messages: list[dict], existing_request_id: str | None = None) -> str | None:
    """Salva o aggiorna il lead nel gestionale. Ritorna requestId o None.
    Se existing_request_id, aggiorna (PUT). Altrimenti crea nuovo (POST).
    """
    conv = [
        {"role": m.get("role", ""), "content": m.get("content", "")[:500]}
        for m in messages
    ]

    payload = {
        "nome": dati.get("nome") or "Cliente",
        "cognome": dati.get("cognome") or "",
        "email": dati.get("email") or "",
        "telefono": dati.get("telefono") or "",
        "tipoAbitazione": dati.get("tipoAbitazione") or "altro",
        "tipoTetto": dati.get("tipoTetto") or "non_so",
        "esposizioneTetto": dati.get("esposizioneTetto") or "non_so",
        "provincia": dati.get("provincia") or "",
        "indirizzo": dati.get("indirizzo") or "",
        "citta": dati.get("citta") or "",
        "superficieMq": dati.get("superficieMq") or 100,
        "numeroPersone": dati.get("numeroPersone") or 3,
        "disponeBolletta": bool(dati.get("disponeBolletta")),
        "spesaMediaMensile": dati.get("spesaMensile"),
        "consumoAnnuoKWh": dati.get("consumoAnnuo"),
        "haSistemaFotovoltaico": bool(dati.get("haFotovoltaico")),
        "interesseBatteriaAccumulo": bool(dati.get("interesseBatteria")),
        "fonte": "telegram",
        "botChatId": str(chat_id),
        "conversazione": conv,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            if existing_request_id:
                payload["requestId"] = existing_request_id
                response = await client.put(
                    GESTIONALE_API,
                    json=payload,
                    headers={"x-api-key": GESTIONALE_API_KEY},
                )
                if response.status_code == 200:
                    log.info(f"[{chat_id}] Lead aggiornato nel gestionale: {existing_request_id}")
                    return existing_request_id
                else:
                    log.error(f"[{chat_id}] Gestionale update errore: {response.status_code} {response.text[:200]}")
                    return None
            else:
                response = await client.post(
                    GESTIONALE_API,
                    json=payload,
                    headers={"x-api-key": GESTIONALE_API_KEY},
                )
                if response.status_code == 201:
                    result = response.json()
                    request_id = result.get("requestId")
                    log.info(f"[{chat_id}] Lead creato nel gestionale: {request_id}")
                    return request_id
                else:
                    log.error(f"[{chat_id}] Gestionale errore: {response.status_code} {response.text[:200]}")
                    return None
    except Exception as e:
        log.error(f"[{chat_id}] Errore salvataggio gestionale: {e}")
        return None


async def update_lead_gestionale(request_id: str, data: dict) -> bool:
    """Aggiorna il lead nel gestionale (analisi, PDF, coordinate)."""
    try:
        payload = {"requestId": request_id, **data}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.put(
                GESTIONALE_API,
                json=payload,
                headers={"x-api-key": GESTIONALE_API_KEY},
            )
            return response.status_code == 200
    except Exception as e:
        log.error(f"Errore aggiornamento gestionale: {e}")
        return False


async def call_analisi(dati: dict, file_data: dict | None = None) -> dict | None:
    """Chiama /api/analisi su homefloo.com con i dati raccolti e eventuale bolletta."""
    form_fields = {
        "requestId": f"telegram-{int(time.time())}",
        "nome": dati.get("nome") or "Cliente",
        "cognome": dati.get("cognome") or "",
        "email": dati.get("email") or "",
        "telefono": dati.get("telefono") or "",
        "tipoAbitazione": dati.get("tipoAbitazione") or "villetta",
        "tipoTetto": dati.get("tipoTetto") or "non_so",
        "esposizioneTetto": dati.get("esposizioneTetto") or "sud",
        "indirizzo": dati.get("indirizzo") or "",
        "citta": dati.get("citta") or "",
        "cap": "",
        "provincia": dati.get("provincia") or "",
        "superficieMq": str(dati.get("superficieMq") or 100),
        "numeroPersone": str(dati.get("numeroPersone") or 3),
        "disponeBolletta": "true" if file_data else "false",
        "spesaMediaMensile": str(dati["spesaMensile"]) if dati.get("spesaMensile") else "",
        "consumoAnnuoKWh": str(dati["consumoAnnuo"]) if dati.get("consumoAnnuo") else "",
        "haSistemaFotovoltaico": str(dati.get("haFotovoltaico") or False).lower(),
        "interesseBatteriaAccumulo": str(dati.get("interesseBatteria") or False).lower(),
    }

    log.info(f"Chiamo analisi con dati: {json.dumps(form_fields, ensure_ascii=False)[:200]}")
    log.info(f"File allegato: {'si' if file_data else 'no'}")

    async with httpx.AsyncClient(timeout=ANALISI_TIMEOUT, follow_redirects=True) as client:
        # Costruisci multipart form data
        files = {}
        if file_data:
            files["file"] = (
                file_data["filename"],
                file_data["bytes"],
                file_data["mime_type"],
            )

        response = await client.post(
            HOMEFLOO_API,
            data=form_fields,
            files=files if files else None,
        )
        if response.status_code != 200:
            log.error(f"Analisi API errore: {response.status_code} {response.text[:200]}")
            return None
        return response.json()


def format_analisi_telegram(analysis: dict) -> str:
    """Formatta il risultato dell'analisi per Telegram."""
    consumi = analysis.get("analisiConsumi", {})
    proposta = analysis.get("propostaImpianto", {})
    fin = analysis.get("analisiFinanziaria", {})
    batteria = proposta.get("batteriaAccumulo", {})
    costo = proposta.get("stimaCosto", {})
    netto = proposta.get("costoNettoStimato", {})

    lines = [
        "ANALISI ENERGETICA HOMEFLOO",
        "=" * 30,
        "",
        "CONSUMI",
        f"  Consumo annuo: {consumi.get('consumoAnnuoStimato', 'N/D')} kWh",
        f"  Spesa annua: {consumi.get('spesaAnnuaStimata', 'N/D')} EUR",
        f"  Profilo: {consumi.get('profiloConsumo', 'N/D')}",
        "",
        "IMPIANTO PROPOSTO",
        f"  Potenza: {proposta.get('potenzaKWp', 'N/D')} kWp",
        f"  Pannelli: {proposta.get('numeroPannelliIndicativo', 'N/D')}",
        f"  Produzione: {proposta.get('produzioneAnnuaStimata', 'N/D')} kWh/anno",
        f"  Autoconsumo: {proposta.get('autoconsumoStimato', 'N/D')}",
    ]

    if batteria.get("consigliata"):
        lines.append(f"  Batteria: {batteria.get('capacitaKWh', 'N/D')} kWh (consigliata)")
    else:
        lines.append("  Batteria: non necessaria")

    lines.extend([
        "",
        "COSTI",
        f"  Lordo: {costo.get('min', 'N/D')} - {costo.get('max', 'N/D')} EUR",
        f"  Netto (con incentivi): {netto.get('min', 'N/D')} - {netto.get('max', 'N/D')} EUR",
        "",
        "RISPARMIO",
        f"  Annuo: {fin.get('risparmioAnnuo', 'N/D')} EUR",
        f"  Rientro: {fin.get('tempoRientro', 'N/D')}",
        f"  In 25 anni: {fin.get('risparmio25Anni', 'N/D')} EUR",
        "",
        "Questa e' una stima indicativa basata sui dati forniti.",
        "Per un preventivo preciso, un nostro consulente ti contatterà.",
    ])

    return "\n".join(lines)


async def download_telegram_file(bot, file_id: str) -> bytes:
    """Scarica un file da Telegram e ritorna i bytes."""
    tg_file = await bot.get_file(file_id)
    bio = BytesIO()
    await tg_file.download_to_memory(bio)
    bio.seek(0)
    return bio.read()


async def extract_bill_header(file_bytes: bytes, mime_type: str) -> dict | None:
    """Estrae nome, cognome, indirizzo dalla bolletta via Claude Haiku.

    Ritorna dict con i campi estratti o None se fallisce.
    """
    if not ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY non configurata — skip estrazione bolletta")
        return None

    b64 = base64.b64encode(file_bytes).decode("utf-8")

    # Per PDF usa document type, per immagini usa image type
    if mime_type == "application/pdf":
        content_block = {
            "type": "document",
            "source": {"type": "base64", "media_type": mime_type, "data": b64},
        }
    else:
        content_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": b64},
        }

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    content_block,
                    {
                        "type": "text",
                        "text": (
                            "Estrai dall'intestazione di questa bolletta elettrica/gas i seguenti dati. "
                            "Rispondi SOLO con un JSON valido, senza altro testo.\n"
                            '{"nome": "...", "cognome": "...", "indirizzo": "...", "citta": "...", "provincia": "..."}\n'
                            "Se un campo non è leggibile, metti null."
                        ),
                    },
                ],
            }
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            if resp.status_code != 200:
                log.warning(f"Claude Haiku errore: {resp.status_code} {resp.text[:200]}")
                return None

            data = resp.json()
            text = data.get("content", [{}])[0].get("text", "")
            # Pulizia: rimuovi eventuale markdown code block
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            log.info(f"Dati estratti dalla bolletta: {result}")
            return result

    except json.JSONDecodeError as e:
        log.warning(f"JSON non valido da Claude Haiku: {e}")
        return None
    except Exception as e:
        log.warning(f"Errore estrazione bolletta: {e}")
        return None


def format_bill_injection(extracted: dict) -> str:
    """Formatta i dati estratti dalla bolletta per l'iniezione nel messaggio a Qwen."""
    parts = []
    nome = extracted.get("nome")
    cognome = extracted.get("cognome")
    if nome or cognome:
        intestatario = f"{nome or ''} {cognome or ''}".strip()
        parts.append(f"- Intestatario: {intestatario}")
    indirizzo = extracted.get("indirizzo")
    if indirizzo:
        parts.append(f"- Indirizzo: {indirizzo}")
    citta = extracted.get("citta")
    provincia = extracted.get("provincia")
    if citta:
        loc = citta
        if provincia:
            loc += f" ({provincia})"
        parts.append(f"- Località: {loc}")

    if not parts:
        return ""

    return (
        "[L'utente ha caricato la bolletta. Dati estratti dalla bolletta:\n"
        + "\n".join(parts) + "\n"
        "Chiedi all'utente se questi dati anagrafici sono corretti per l'analisi. "
        "Se confermati, NON chiederli di nuovo. Se l'utente dice che sono diversi, chiedi quelli giusti.\n"
        "Spesa e consumo verranno estratti automaticamente — NON chiederli.]"
    )


# --- Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start — reset sessione e benvenuto."""
    chat_id = update.effective_chat.id
    sessions[chat_id] = {"messages": [], "last_active": time.time(), "file": None, "request_id": None}
    save_sessions()
    log_conversation(chat_id, "system", "START")

    welcome = (
        "Ciao! Sono l'assistente virtuale di Homefloo.\n\n"
        "Posso aiutarti a capire se il fotovoltaico fa per te "
        "e stimare quanto potresti risparmiare.\n\n"
        "Raccontami: stai pensando al fotovoltaico? "
        "Hai una bolletta sotto mano?"
    )
    await update.message.reply_text(welcome)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /reset — nuova conversazione."""
    chat_id = update.effective_chat.id
    sessions[chat_id] = {"messages": [], "last_active": time.time(), "file": None, "request_id": None}
    save_sessions()
    log_conversation(chat_id, "system", "RESET")
    await update.message.reply_text("Conversazione azzerata. Come posso aiutarti?")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gestisce foto — salva in sessione come bolletta."""
    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    # Prendi la foto piu grande
    photo = update.message.photo[-1]

    try:
        file_bytes = await download_telegram_file(context.bot, photo.file_id)
        session["file"] = {
            "bytes": file_bytes,
            "mime_type": "image/jpeg",
            "filename": "bolletta.jpg",
        }
        log.info(f"[{chat_id}] Foto bolletta salvata: {len(file_bytes)} bytes")
        log_conversation(chat_id, "user", "[FOTO BOLLETTA]", extra={"file_size": len(file_bytes)})

        # Estrai dati anagrafici dalla bolletta via Claude Haiku
        await update.effective_chat.send_action("typing")
        extracted = await extract_bill_header(file_bytes, "image/jpeg")

        # Costruisci messaggio per Qwen con dati estratti (o fallback generico)
        if extracted:
            injection = format_bill_injection(extracted)
            log_conversation(chat_id, "system", "BILL_EXTRACTED", extra={"data": extracted})
        else:
            injection = ""

        if injection:
            user_msg = injection
        else:
            user_msg = (
                "[L'utente ha caricato una foto della bolletta. La bolletta verra analizzata automaticamente "
                "— NON chiedere spesa mensile ne consumo annuo, quei dati verranno estratti dalla bolletta. "
                "Conferma la ricezione e continua a raccogliere gli ALTRI dati necessari (indirizzo, tipo abitazione, tetto, ecc.).]"
            )

        messages = session["messages"]
        messages.append({"role": "user", "content": user_msg})
        messages = trim_session(messages)
        session["messages"] = messages

        await update.effective_chat.send_action("typing")
        response_text = await call_claude(messages)
        clean_text, _ = extract_dati_completi(response_text)

        messages.append({"role": "assistant", "content": clean_text})
        session["messages"] = messages

        log.info(f"[{chat_id}] Bot: {clean_text[:80]}")
        await update.message.reply_text(clean_text)

    except Exception as e:
        log.error(f"[{chat_id}] Errore foto: {e}")
        await update.message.reply_text(
            "Ho ricevuto la foto ma c'e stato un problema tecnico. Puoi riprovare?"
        )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gestisce documenti (PDF bolletta)."""
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    doc = update.message.document

    if not doc:
        return

    mime = doc.mime_type or ""
    if mime not in ("application/pdf", "image/jpeg", "image/png", "image/webp"):
        await update.message.reply_text(
            "Accetto bollette in formato PDF o immagine (JPG, PNG). Puoi riprovare?"
        )
        return

    try:
        file_bytes = await download_telegram_file(context.bot, doc.file_id)
        session["file"] = {
            "bytes": file_bytes,
            "mime_type": mime,
            "filename": doc.file_name or "bolletta",
        }
        log.info(f"[{chat_id}] Documento salvato: {doc.file_name} ({len(file_bytes)} bytes, {mime})")
        log_conversation(chat_id, "user", f"[DOCUMENTO: {doc.file_name}]", extra={"file_size": len(file_bytes), "mime": mime})

        # Estrai dati anagrafici dalla bolletta via Claude Haiku
        await update.effective_chat.send_action("typing")
        extracted = await extract_bill_header(file_bytes, mime)

        if extracted:
            injection = format_bill_injection(extracted)
            log_conversation(chat_id, "system", "BILL_EXTRACTED", extra={"data": extracted})
        else:
            injection = ""

        if injection:
            user_msg = injection
        else:
            user_msg = (
                f"[L'utente ha caricato la bolletta come {mime.split('/')[-1].upper()}. La bolletta verra analizzata automaticamente "
                "— NON chiedere spesa mensile ne consumo annuo, quei dati verranno estratti dalla bolletta. "
                "Conferma la ricezione e continua a raccogliere gli ALTRI dati necessari (indirizzo, tipo abitazione, tetto, ecc.).]"
            )

        messages = session["messages"]
        messages.append({"role": "user", "content": user_msg})
        messages = trim_session(messages)
        session["messages"] = messages

        await update.effective_chat.send_action("typing")
        response_text = await call_claude(messages)
        clean_text, _ = extract_dati_completi(response_text)

        messages.append({"role": "assistant", "content": clean_text})
        session["messages"] = messages

        log.info(f"[{chat_id}] Bot: {clean_text[:80]}")
        await update.message.reply_text(clean_text)

    except Exception as e:
        log.error(f"[{chat_id}] Errore documento: {e}")
        await update.message.reply_text(
            "Ho ricevuto il file ma c'e stato un problema. Puoi riprovare?"
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gestisce messaggi di testo — cuore del bot."""
    chat_id = update.effective_chat.id
    user_text = update.message.text

    if not user_text:
        return

    log.info(f"[{chat_id}] User: {user_text[:80]}")
    log_conversation(chat_id, "user", user_text)

    session = get_session(chat_id)
    messages = session["messages"]
    messages.append({"role": "user", "content": user_text})
    messages = trim_session(messages)
    session["messages"] = messages

    # Typing indicator
    await update.effective_chat.send_action("typing")

    try:
        # Chiama Claude Haiku
        response_text = await call_claude(messages)

        # Controlla se Claude ha segnalato dati completi
        clean_text, dati = extract_dati_completi(response_text)

        # Salva risposta pulita nella sessione
        messages.append({"role": "assistant", "content": clean_text})
        session["messages"] = messages
        save_sessions()

        log.info(f"[{chat_id}] Bot: {clean_text[:80]}")
        log_conversation(chat_id, "assistant", clean_text)

        # Manda il riepilogo al cliente
        await update.message.reply_text(clean_text)

        # Se i dati sono completi, salva il lead e lancia l'analisi
        if dati:
            log.info(f"[{chat_id}] DATI COMPLETI ricevuti: {json.dumps(dati, ensure_ascii=False)[:200]}")
            log_conversation(chat_id, "system", "DATI_COMPLETI", extra={"dati": dati})

            # STEP 1: Cerca lead esistente o crea nuovo
            existing_id = session.get("request_id")
            if not existing_id:
                existing_lead = await find_existing_lead(chat_id)
                if existing_lead and existing_lead.get("stato") in ("NUOVA", None):
                    existing_id = existing_lead.get("requestId")
                    log.info(f"[{chat_id}] Trovato lead esistente da aggiornare: {existing_id}")

            lead_request_id = await save_lead_gestionale(dati, chat_id, messages, existing_id)
            if lead_request_id:
                session["request_id"] = lead_request_id
                save_sessions()
                log_conversation(chat_id, "system", "LEAD_SALVATO", extra={"requestId": lead_request_id, "updated": bool(existing_id)})

            await update.effective_chat.send_action("typing")
            await update.message.reply_text(
                "Sto preparando la tua analisi energetica personalizzata... ci vorra circa un minuto."
            )

            # STEP 2: Lancia l'analisi Claude (se fallisce, il lead c'e comunque)
            try:
                file_data = session.get("file")
                result = await call_analisi(dati, file_data)

                if result and result.get("success"):
                    analysis = result.get("analysis", {})
                    pdf_b64 = result.get("reportPdfBase64")
                    lat = result.get("latitudine")
                    lng = result.get("longitudine")

                    # Aggiorna il lead nel gestionale con analisi + PDF + coordinate
                    if lead_request_id:
                        update_data = {
                            "analisiEnergetica": analysis,
                            "stato": "COMPLETATA",
                        }
                        if lat is not None:
                            update_data["latitudine"] = lat
                        if lng is not None:
                            update_data["longitudine"] = lng
                        if pdf_b64:
                            nome = dati.get("nome") or "Cliente"
                            update_data["reportPdf"] = {
                                "base64": pdf_b64,
                                "mimeType": "application/pdf",
                                "fileName": f"Report-{nome}-{dati.get('cognome', '')}.pdf",
                            }
                        await update_lead_gestionale(lead_request_id, update_data)

                    # Manda il report testuale
                    report = format_analisi_telegram(analysis)
                    await update.message.reply_text(report)

                    # Manda il PDF se disponibile
                    if pdf_b64:
                        try:
                            pdf_bytes = base64.b64decode(pdf_b64)
                            nome = dati.get("nome") or "Cliente"
                            filename = f"Analisi-Energetica-{nome}.pdf"
                            await update.message.reply_document(
                                document=BytesIO(pdf_bytes),
                                filename=filename,
                                caption="Ecco il tuo report personalizzato in PDF.",
                            )
                            log.info(f"[{chat_id}] PDF inviato: {filename} ({len(pdf_bytes)} bytes)")
                        except Exception as pdf_err:
                            log.error(f"[{chat_id}] Errore invio PDF: {pdf_err}")

                    log.info(f"[{chat_id}] Analisi inviata con successo")
                    log_conversation(chat_id, "system", "ANALISI_OK", extra={"requestId": lead_request_id or result.get("requestId")})
                else:
                    error = result.get("error", "Errore sconosciuto") if result else "Nessuna risposta"
                    log.error(f"[{chat_id}] Analisi fallita: {error}")
                    log_conversation(chat_id, "system", "ANALISI_FALLITA", extra={"error": error})
                    # Aggiorna il lead con nota di errore
                    if lead_request_id:
                        await update_lead_gestionale(lead_request_id, {
                            "note": f"Analisi fallita: {error}",
                        })
                    await update.message.reply_text(
                        "Mi dispiace, c'e stato un problema con l'analisi. "
                        "Un nostro consulente ti contatterà per completarla di persona."
                    )
            except Exception as e:
                log.error(f"[{chat_id}] Errore analisi: {e}")
                log_conversation(chat_id, "system", "ANALISI_ERRORE", extra={"error": str(e)})
                if lead_request_id:
                    await update_lead_gestionale(lead_request_id, {
                        "note": f"Errore tecnico: {str(e)[:200]}",
                    })
                await update.message.reply_text(
                    "Mi dispiace, c'e stato un problema tecnico con l'analisi. "
                    "Un nostro consulente ti contatterà."
                )

    except httpx.TimeoutException:
        log.warning(f"[{chat_id}] Timeout da Claude")
        messages.pop()
        await update.message.reply_text(
            "C'e stato un problema di connessione. Riprova tra qualche secondo."
        )

    except Exception as e:
        log.error(f"[{chat_id}] Errore: {e}")
        messages.pop()
        await update.message.reply_text(
            "C'e stato un problema tecnico. Riprova tra un momento."
        )


def main():
    """Avvia il bot."""
    if not TELEGRAM_TOKEN:
        raise RuntimeError("HOMEFLOO_TELEGRAM_TOKEN env var is required")
    if not GESTIONALE_API_KEY:
        raise RuntimeError("GESTIONALE_API_KEY env var is required")
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY env var is required")

    log.info("Avvio Homefloo Telegram Bot...")
    log.info(f"Chat model: {CHAT_MODEL}")
    log.info(f"Analisi API: {HOMEFLOO_API}")
    log.info(f"Gestionale API: {GESTIONALE_API}")
    log.info(f"Data dir: {DATA_DIR}")
    load_sessions()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Comandi
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))

    # Foto (bolletta come immagine)
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Documenti (bolletta come PDF)
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Messaggi di testo
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Bot in ascolto...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
