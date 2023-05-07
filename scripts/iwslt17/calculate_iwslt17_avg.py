results = r"26.4 & 27.5 &37.4 &39.9 & 32.9 & 34.9 & 27.6 & 33.8"
results = results.replace(" ", "").replace(r"\bf", "").split("&")
e2x = [float(results[i]) for i in range(0, len(results), 2)]
x2e = [float(results[i]) for i in range(1, len(results), 2)]
x2x = e2x + x2e
e2x_avg = sum(e2x) / len(e2x)
x2e_avg = sum(x2e) / len(x2e)
x2x_avg = sum(x2x) / len(x2x)
print(f"e2x: {e2x_avg}")
print(f"x2e: {x2e_avg}")
print(f"x2x: {x2x_avg}")
