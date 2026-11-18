"""Report notifications: Telegram or WhatsApp.

Providers (``WHATSAPP_PROVIDER``, named so for history):

* ``telegram`` - free and reliable, Telegram's official Bot API. Create a bot
  with @BotFather (it gives you TELEGRAM_BOT_TOKEN), send your bot any message,
  then run ``python automate.py telegram-chat-id`` to get TELEGRAM_CHAT_ID.
* ``callmebot`` - free for personal use. Save +34 611 021 695 in your contacts,
  send it "I allow callmebot to send me messages" on WhatsApp, and it replies
  with your API key. Set WHATSAPP_PHONE (with country code) and CALLMEBOT_API_KEY.
* ``twilio`` - paid, for a business sender: TWILIO_ACCOUNT_SID,
  TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM and WHATSAPP_PHONE.
* empty - the message is only written to the log.

Long reports are split into several messages. A failed send never fails a run.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

from config.settings import Settings

logger = logging.getLogger(__name__)

CALLMEBOT_URL = "https://api.callmebot.com/whatsapp.php"
TWILIO_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
TELEGRAM_URL = "https://api.telegram.org/bot{token}/{method}"
CHUNK_CHARS = 1000
TELEGRAM_CHUNK_CHARS = 3500   # Telegram allows 4096 per message


def split_message(text: str, limit: int = CHUNK_CHARS) -> List[str]:
    """Split on line breaks into pieces of at most ``limit`` characters."""
    chunks, current = [], ""
    for line in text.splitlines():
        while len(line) > limit:  # a single very long line
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return chunks


class WhatsAppNotifier:
    def __init__(
        self,
        settings: Settings,
        *,
        get: Optional[Callable] = None,
        post: Optional[Callable] = None,
        pause: float = 2.0,
    ) -> None:
        self.settings = settings
        self._get = get
        self._post = post
        self.pause = pause

    @property
    def configured(self) -> bool:
        s = self.settings
        if s.whatsapp_provider == "callmebot":
            return bool(s.whatsapp_phone and s.callmebot_api_key)
        if s.whatsapp_provider == "telegram":
            return bool(s.telegram_bot_token and s.telegram_chat_id)
        if s.whatsapp_provider == "twilio":
            return bool(s.whatsapp_phone and s.twilio_account_sid and s.twilio_auth_token and s.twilio_whatsapp_from)
        return False

    def send(self, text: str) -> bool:
        """Send ``text``; returns True if every piece was accepted."""
        if not self.configured:
            logger.info("Notifications not configured; report follows:\n%s", text)
            return False
        ok = True
        limit = TELEGRAM_CHUNK_CHARS if self.settings.whatsapp_provider == "telegram" else CHUNK_CHARS
        chunks = split_message(text, limit)
        for i, chunk in enumerate(chunks, start=1):
            body = f"({i}/{len(chunks)})\n{chunk}" if len(chunks) > 1 else chunk
            try:
                ok = self._send_one(body) and ok
            except Exception as exc:  # noqa: BLE001 - never fail the run on a notification
                # Only the exception type: messages can contain the bot token in a URL.
                logger.warning("Notification send failed (%s).", type(exc).__name__)
                ok = False
            if i < len(chunks) and self.pause:
                time.sleep(self.pause)
        logger.info("%s report sent in %d message(s)%s.", self.settings.whatsapp_provider,
                    len(chunks), "" if ok else " (with errors)")
        return ok

    def _send_one(self, body: str) -> bool:
        s = self.settings
        if s.whatsapp_provider == "telegram":
            post = self._post or _requests().post
            url = TELEGRAM_URL.format(token=s.telegram_bot_token, method="sendMessage")
            payload = {"chat_id": s.telegram_chat_id, "text": body, "parse_mode": "Markdown",
                       "disable_web_page_preview": True}
            resp = post(url, json=payload, timeout=30)
            if getattr(resp, "status_code", 0) == 400:  # formatting rejected: send as plain text
                payload.pop("parse_mode")
                resp = post(url, json=payload, timeout=30)
        elif s.whatsapp_provider == "callmebot":
            get = self._get or _requests().get
            resp = get(
                CALLMEBOT_URL,
                params={"phone": s.whatsapp_phone, "text": body, "apikey": s.callmebot_api_key},
                timeout=30,
            )
        else:
            post = self._post or _requests().post
            to = s.whatsapp_phone if s.whatsapp_phone.startswith("whatsapp:") else f"whatsapp:{s.whatsapp_phone}"
            sender = s.twilio_whatsapp_from
            sender = sender if sender.startswith("whatsapp:") else f"whatsapp:{sender}"
            resp = post(
                TWILIO_URL.format(sid=s.twilio_account_sid),
                data={"From": sender, "To": to, "Body": body},
                auth=(s.twilio_account_sid, s.twilio_auth_token),
                timeout=30,
            )
        status = getattr(resp, "status_code", 0)
        if status >= 400:
            logger.warning("Notification provider returned HTTP %s.", status)
            return False
        return True


def telegram_chat_ids(token: str, *, get: Optional[Callable] = None) -> List[Tuple[str, str]]:
    """(chat id, name) of everyone who has messaged the bot recently."""
    get = get or _requests().get
    resp = get(TELEGRAM_URL.format(token=token, method="getUpdates"), timeout=30)
    if getattr(resp, "status_code", 0) != 200:
        raise RuntimeError(f"Telegram returned HTTP {getattr(resp, 'status_code', '?')}; check TELEGRAM_BOT_TOKEN.")
    found: Dict[str, str] = {}
    for update in resp.json().get("result", []):
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        if "id" in chat:
            name = chat.get("username") or " ".join(
                part for part in (chat.get("first_name"), chat.get("last_name")) if part
            ) or chat.get("title", "")
            found[str(chat["id"])] = name
    return list(found.items())


def _requests():
    import requests

    return requests


Notifier = WhatsAppNotifier

__all__ = ["Notifier", "WhatsAppNotifier", "split_message", "telegram_chat_ids"]
