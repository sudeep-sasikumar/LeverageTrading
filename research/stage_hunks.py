"""Developer helper: list or stage individual diff hunks (git add -p without the prompt).

  python -m research.stage_hunks list lts/service.py
  python -m research.stage_hunks add lts/service.py 0 2 5
"""
import subprocess
import sys


def hunks(path):
    d = subprocess.run(["git", "diff", "-U3", "--", path], capture_output=True).stdout.decode("utf-8")
    if not d:
        return "", []
    head, rest = d.split("\n@@", 1)
    parts = ("@@" + rest).split("\n@@")
    return head + "\n", ["@@" + p.lstrip("@") if not p.startswith("@@") else p for p in parts]


def main():
    cmd, path, *idx = sys.argv[1:]
    head, hs = hunks(path)
    if cmd == "list":
        for i, h in enumerate(hs):
            lines = h.splitlines()
            ch = [l for l in lines[1:] if l[:1] in "+-"][:3]
            print(f"[{i}] {lines[0][:60]}  " + " | ".join(x[:60] for x in ch))
        return
    keep = [hs[int(i)] for i in idx]
    patch = head + "\n".join(h.rstrip("\n") for h in keep) + "\n"
    r = subprocess.run(["git", "apply", "--cached", "--recount", "-"], input=patch.encode("utf-8"), capture_output=True)
    if r.returncode:
        raise SystemExit(r.stderr.decode("utf-8", "replace"))
    print(f"staged {len(keep)} hunk(s) of {path}")


if __name__ == "__main__":
    main()
