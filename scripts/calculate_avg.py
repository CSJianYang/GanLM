result=r"\bf 87.3 & \bf 78.3 &     82.7 & \bf 83.1 &     82.2 &     83.8 &     83.3 & \bf 77.3 &     81.3 &     73.1 & \bf 80.3 & \bf 79.9 &     71.2 &     81.3 & \bf 81.8"
result=result.replace(r"\bf", "").replace(" ", "").split("&")
result=[float(s) for s in result]
avg = sum(result) / len(result)
print(avg)
