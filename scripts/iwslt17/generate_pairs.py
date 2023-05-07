LANGS = "de en it nl ro".split()
PAIRS = []
for src in LANGS:
    for tgt in LANGS:
        if src != tgt:
            PAIRS.append(f"\"{src}-{tgt}\"")

print(",".join(PAIRS))
