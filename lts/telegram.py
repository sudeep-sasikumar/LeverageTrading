"""Telegram notifier and command listener (Bot API over HTTPS, no extra dependency).

Without TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID the notifier runs in dry-run mode and
only writes messages to the log.
"""
from __future__ import annotations

import asyncio
import html
import logging

import aiohttp

from .config import Config

log = logging.getLogger("lts.telegram")


def esc(text) -> str:
    return html.escape(str(text), quote=False)


class Telegram:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.token = cfg.telegram_token
        self.chat_id = str(cfg.telegram_chat_id)
        self.enabled = bool(cfg.telegram.enabled and self.token and self.chat_id)
        self._session: aiohttp.ClientSession | None = None
        self._offset = 0
        if not self.enabled:
            log.warning("Telegram not configured - running in dry-run mode (messages go to the log)")

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40))
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _call(self, method: str, payload: dict) -> dict | None:
        s = await self._sess()
        async with s.post(f"https://api.telegram.org/bot{self.token}/{method}", json=payload) as r:
            data = await r.json(content_type=None)
            if not data.get("ok"):
                log.warning("telegram %s failed: %s", method, str(data)[:200])
                return None
            return data

    async def send(self, text: str) -> bool:
        """HTML-formatted message to the configured chat. Never raises."""
        if not self.enabled:
            log.info("[telegram dry-run]\n%s", text)
            return False
        for attempt in range(3):
            try:
                ok = await self._call("sendMessage", {"chat_id": self.chat_id, "text": text[:4000],
                                                      "parse_mode": "HTML", "disable_web_page_preview": True})
                if ok:
                    return True
            except Exception as e:
                log.warning("telegram send error: %s", e)
            await asyncio.sleep(2 * (attempt + 1))
        return False

    async def send_once(self, text: str) -> None:
        try:
            sent = await self.send(text)
            print("sent" if sent else "not sent (dry-run or error - see log)")
        finally:
            await self.close()

    async def poll_commands(self, handler) -> None:
        """Long-poll for commands from the configured chat; handler(text) -> reply text."""
        if not self.enabled:
            return
        while True:
            try:
                data = await self._call("getUpdates", {"offset": self._offset, "timeout": 25,
                                                       "allowed_updates": ["message"]})
                for upd in (data or {}).get("result", []):
                    self._offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    if str((msg.get("chat") or {}).get("id")) != self.chat_id:
                        continue                      # ignore anyone but the owner chat
                    text = (msg.get("text") or "").strip()
                    if text.startswith("/"):
                        reply = await handler(text)
                        if reply:
                            await self.send(reply)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("telegram poll error: %s", e)
                await asyncio.sleep(10)
