xnli="86.3&76.7  &79.7 &80.8   &79.7 &81.6 &82.0 &74.6 &78.6 &70.8 &77.4 &77.1 &65.3 &79.3 &79.3"
xnli=xnli.replace(" ", "").replace("&", " ").split()
xnli=[float(s) for s in xnli]
avg=sum(xnli)/len(xnli)
print(avg)