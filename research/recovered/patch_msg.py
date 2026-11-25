import pathlib

ROOT = pathlib.Path(r"D:\Leverage trading")


def sub(path, pairs):
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path, a[:70])
        s = s.replace(a, b)
    p.write_text(s, encoding="utf-8")


sub("lts/config.py", [(
    "    account_size_usdt: float = 0.0       # optional: lets the bot quote margin in USDT\n",
    "    account_size_usdt: float = 0.0       # optional: lets the bot quote margin in USDT\n"
    "    local_utc_offset_minutes: int = 330  # times are shown in UTC and in this local zone\n"
    "    local_tz_label: str = \"IST\"\n")])

sub("lts/signals.py", [
    ('''                  f"Time stop: close at market by {utc(sig['expires_ts'])} if neither level is hit",''',
     '''                  f"Time stop: close at market by {utc(sig['expires_ts'])} "
                  f"({utc(sig['expires_ts'] + cfg.telegram.local_utc_offset_minutes * 60_000, '%H:%M')} "
                  f"{esc(cfg.telegram.local_tz_label)}) if neither level is hit",'''),
    ('''            lines.append(f"P(target first) {sig['p']:.0%} · EV {sig['ev_r']:+.2f}R · {esc(sig['model_version'])}")''',
     '''            label = "Unvalidated model estimate" if sig["grade"] == "P" else "Model"
            lines.append(f"{label}: P(target first) {sig['p']:.0%} · EV {sig['ev_r']:+.2f}R · "
                         f"{esc(sig['model_version'])}")'''),
])
sub("config.yaml", [(
    "  account_size_usdt: 0          # optional: set to quote margin in USDT in each signal\n",
    "  account_size_usdt: 0          # optional: set to quote margin in USDT in each signal\n"
    "  local_utc_offset_minutes: 330 # show times in IST as well as UTC\n"
    "  local_tz_label: IST\n")])
print("patched")
