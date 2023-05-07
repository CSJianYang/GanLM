import sys


for line in sys.stdin:
    line = line.replace(" ", "").replace("\u2581", " ").strip()
    sys.stdout.write(line+"\n")
