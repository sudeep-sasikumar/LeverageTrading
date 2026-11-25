import pathlib, re

ROOT = pathlib.Path(r"D:\Leverage trading")
p = ROOT / "README.md"
s = p.read_text(encoding="utf-8")
start = s.index("## Deploy on a Hostinger VPS (Ubuntu)")
end = s.index("## Telegram")
new = """## Running on the VPS (Hostinger Docker Manager)

Same pattern as CryptoLaunchScanner. Pick a VPS region Binance serves (not the US).

1. Every push to `main` runs the tests, builds the image and publishes
   `ghcr.io/sudeep-sasikumar/leveragetrading:latest` (GitHub Actions).
2. On GitHub: your profile → Packages → `leveragetrading` → Package settings → make it
   public, so the VPS can pull it without a login. Needed once.
3. Hostinger → VPS → Docker Manager → Compose → Compose from URL:
   `https://raw.githubusercontent.com/sudeep-sasikumar/LeverageTrading/main/docker-compose.yaml`,
   project name `leverage-trading`.
4. Enter the environment variables, then Deploy:

   | Variable | Needed | What |
   |---|---|---|
   | `TELEGRAM_BOT_TOKEN` | yes | from @BotFather. For `/status` etc. use a bot no other app polls |
   | `TELEGRAM_CHAT_ID` | yes | your chat id; a group id starts with `-100` |
   | `ACCOUNT_SIZE_USDT` | optional | quotes each signal's margin in USDT |
   | `RISK_PER_TRADE` | optional | account fraction lost at a stop (default `0.01`) |
   | `MAX_SIGNALS_PER_DAY` | optional | default `3` |
   | `BACKFILL_DAYS` | optional | history downloaded on first start (default `30`) |

The first start on an empty volume downloads 30 days of history, rebuilds the labelled
candidates and trains the first model (about 40 minutes), then sends "LTS started" on
Telegram. Data (database, models, logs) lives in the `lts_data` volume and survives
redeploys; a redeploy pulls the newest image and resumes within a minute.

No ports are opened: the bot only makes outbound calls.

"""
s = s[:start] + new + s[end:]
s = s.replace("cp .env.example .env                               # add Telegram token + chat id",
              "cp .env.example .env                               # add Telegram token + chat id (local runs only)")
p.write_text(s, encoding="utf-8")

p = ROOT / "docs" / "DESIGN.md"
s = p.read_text(encoding="utf-8")
a = """- Deployment: Ubuntu VPS, systemd service, `.env` for the Telegram token. The VPS must
  be in a region Binance serves (not the US).
- Secrets never enter the repo; `.env` is git-ignored."""
b = """- Deployment: GitHub Actions builds a Docker image to GHCR on every push; Hostinger
  Docker Manager deploys `docker-compose.yaml` from its raw URL. The Telegram token and
  chat id are environment variables set in Docker Manager. Data lives in a named volume.
  The VPS must be in a region Binance serves (not the US).
- Secrets never enter the repo or the image; `.env` is git-ignored and docker-ignored."""
assert a in s
p.write_text(s.replace(a, b), encoding="utf-8")

p = ROOT / ".env.example"
p.write_text("""# For running on this PC: copy to .env (git-ignored) and fill in. Never commit real values.
# On the VPS these are entered as environment variables in Hostinger Docker Manager instead.
# 1. Create a bot with @BotFather in Telegram and paste its token here.
TELEGRAM_BOT_TOKEN=
# 2. Send any message to your bot, then open
#    https://api.telegram.org/bot<TOKEN>/getUpdates and copy result[0].message.chat.id
TELEGRAM_CHAT_ID=
# Optional
ACCOUNT_SIZE_USDT=
RISK_PER_TRADE=
MAX_SIGNALS_PER_DAY=
""", encoding="utf-8")
print("ok")
