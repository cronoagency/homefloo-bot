"""
Homefloo Telegram Bot — Ponte tra Telegram e endpoint Modal.

Qwen 3.5 9B fa la conversazione, questo script fa da ponte.
Tiene la storia per ogni utente (multi-turno).
Quando Qwen segnala dati completi, chiama /api/analisi per il report.
Supporta foto e PDF bolletta — li tiene in sessione per l'analisi.

Config via env vars:
    HOMEFLOO_TELEGRAM_TOKEN (required)
    GESTIONALE_API_KEY (required)
    MODAL_ENDPOINT (optional)
    HOMEFLOO_API (optional)
    GESTIONALE_API (optional)
    DATA_DIR (optional, default /app/data)
"""

import asyncio
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
MODAL_ENDPOINT = os.environ.get(
    "MODAL_ENDPOINT",
    "https://cronoagency--homefloo-bot-homefloobot-chat.modal.run",
)
MODAL_HEALTH_ENDPOINT = MODAL_ENDPOINT.replace("-chat.", "-health.")
HOMEFLOO_API = os.environ.get(
    "HOMEFLOO_API",
    "http://gcgoo88g4ow8sccokc0o48c4.91.98.89.69.sslip.io/api/analisi",
)
GESTIONALE_API = os.environ.get(
    "GESTIONALE_API",
    "https://gestionale.homefloo.com/api/analisi-energetiche/bot",
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
SESSIONS_PATH = DATA_DIR / "sessions.json"
CONV_LOG_PATH = DATA_DIR / "conversations.jsonl"

MAX_HISTORY = 20  # max messaggi per sessione (10 turni user+assistant)
SESSION_EXPIRE_HOURS = 48  # sessioni piu vecchie di 48h vengono scartate
REQUEST_TIMEOUT = 120  # secondi (cold start puo essere 35s)
ANALISI_TIMEOUT = 180  # secondi (Claude Sonnet per analisi)
KEEPWARM_INTERVAL = 240  # secondi (4 minuti)
KEEPWARM_START_HOUR = 8
KEEPWARM_END_HOUR = 21

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


async def call_modal(messages: list[dict]) -> str:
    """Chiama l'endpoint Modal con la conversazione."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        response = await client.post(
            MODAL_ENDPOINT,
            json={"messages": messages},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Mi dispiace, non sono riuscito a generare una risposta.")


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
    """Aggiorna il lead nel gestionale (es. con risultato analisi)."""
    try:
        payload = {"requestId": request_id, **data}
        async with httpx.AsyncClient(timeout=30) as client:
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

        # Di a Qwen che l'utente ha mandato la bolletta
        messages = session["messages"]
        messages.append({"role": "user", "content": "[L'utente ha caricato una foto della bolletta. La bolletta verra analizzata automaticamente — NON chiedere spesa mensile ne consumo annuo, quei dati verranno estratti dalla bolletta. Conferma la ricezione e continua a raccogliere gli ALTRI dati necessari (indirizzo, tipo abitazione, tetto, ecc.).]"})
        messages = trim_session(messages)
        session["messages"] = messages

        await update.effective_chat.send_action("typing")
        response_text = await call_modal(messages)
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

        messages = session["messages"]
        messages.append({"role": "user", "content": f"[L'utente ha caricato la bolletta come {mime.split('/')[-1].upper()}. La bolletta verra analizzata automaticamente — NON chiedere spesa mensile ne consumo annuo, quei dati verranno estratti dalla bolletta. Conferma la ricezione e continua a raccogliere gli ALTRI dati necessari (indirizzo, tipo abitazione, tetto, ecc.).]"})
        messages = trim_session(messages)
        session["messages"] = messages

        await update.effective_chat.send_action("typing")
        response_text = await call_modal(messages)
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
        # Chiama Modal
        response_text = await call_modal(messages)

        # Controlla se Qwen ha segnalato dati completi
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

                    # Aggiorna il lead nel gestionale con il risultato
                    if lead_request_id:
                        await update_lead_gestionale(lead_request_id, {
                            "analisiEnergetica": analysis,
                            "stato": "COMPLETATA",
                        })

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
        log.warning(f"[{chat_id}] Timeout da Modal")
        messages.pop()
        await update.message.reply_text(
            "Mi sto svegliando, dammi qualche secondo... Riprova tra poco!"
        )

    except Exception as e:
        log.error(f"[{chat_id}] Errore: {e}")
        messages.pop()
        await update.message.reply_text(
            "C'e stato un problema tecnico. Riprova tra un momento."
        )


async def keepwarm_loop():
    """Pinga Modal health endpoint ogni 4 min (8-21) per evitare cold start."""
    import datetime
    log.info(f"Keepwarm loop avviato (ogni {KEEPWARM_INTERVAL}s, ore {KEEPWARM_START_HOUR}-{KEEPWARM_END_HOUR})")
    while True:
        try:
            await asyncio.sleep(KEEPWARM_INTERVAL)
            hour = datetime.datetime.now().hour
            if hour < KEEPWARM_START_HOUR or hour >= KEEPWARM_END_HOUR:
                continue
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(MODAL_HEALTH_ENDPOINT)
                if resp.status_code != 200:
                    log.warning(f"Keepwarm FAIL: HTTP {resp.status_code}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning(f"Keepwarm error: {e}")


def main():
    """Avvia il bot."""
    if not TELEGRAM_TOKEN:
        raise RuntimeError("HOMEFLOO_TELEGRAM_TOKEN env var is required")
    if not GESTIONALE_API_KEY:
        raise RuntimeError("GESTIONALE_API_KEY env var is required")

    log.info("Avvio Homefloo Telegram Bot...")
    log.info(f"Endpoint Modal: {MODAL_ENDPOINT}")
    log.info(f"Analisi API: {HOMEFLOO_API}")
    log.info(f"Gestionale API: {GESTIONALE_API}")
    log.info(f"Data dir: {DATA_DIR}")
    load_sessions()

    async def post_init(application):
        application.keepwarm_task = asyncio.create_task(keepwarm_loop())

    async def post_shutdown(application):
        if hasattr(application, "keepwarm_task"):
            application.keepwarm_task.cancel()

    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).post_shutdown(post_shutdown).build()

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
