import matplotlib.pyplot as plt

cols = ["h", "N (mine)", "dt (mine)", "E (mine)", "time (mine)",
        "N (pyTDGL)", "dt (pyTDGL)", "E (pyTDGL)", "time (pyTDGL)"]

rows = [
    ["2.0",  "256",   "0.05", "7.68e-03", "0.60 s",  "731",   "0.05",   "6.38e-03", "0.74 s"],
    ["1.0",  "961",   "0.05", "1.64e-03", "1.57 s",  "3134",  "0.05",   "2.15e-03", "2.07 s"],
    ["0.5",  "3721",  "0.05", "3.96e-04", "5.29 s",  "12902", "0.05",   "1.77e-03", "8.97 s"],
    ["0.25", "14641", "0.05", "9.81e-05", "21.54 s", "49879", "0.0125", "6.82e-04", "204.31 s"],
]

fig, ax = plt.subplots(figsize=(12, 2.2))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
ax.set_title("Uniform current state, 30x30 strip, J = 0.25, gamma = 0, tmax = 200, 1 thread", pad=12)
plt.savefig("comparison_table.pdf", bbox_inches="tight")