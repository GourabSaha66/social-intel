"""Draws docs/architecture.png. Run: python scripts/make_architecture.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)
ax.axis("off")


def box(x, y, w, h, title, sub="", color="#dbe9f6"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=color, ec="#33506b", lw=1.2))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center",
            fontsize=10, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8, color="#333")


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#33506b", lw=1.4))


# sources
box(0.2, 4.6, 2.0, 0.9, "Reddit API (off)", "needs approval", "#eeeeee")
box(0.2, 3.3, 2.0, 0.9, "YouTube API", "Data API v3", "#fde4cf")
box(0.2, 2.0, 2.0, 0.9, "LIAR dataset", "labels for misinfo", "#e6f2d9")
# raw + clean
box(3.0, 3.95, 1.9, 1.0, "Raw JSON", "data/raw", "#eeeeee")
box(5.4, 3.95, 2.1, 1.0, "Normalize + Clean", "emoji, langid, dedupe")
# enrichment
box(8.1, 5.2, 2.2, 0.85, "Sentiment", "XLM-R (multilingual)")
box(8.1, 4.1, 2.2, 0.85, "Topics", "BERTopic")
box(8.1, 3.0, 2.2, 0.85, "Graph", "NetworkX + Neo4j")
box(8.1, 1.9, 2.2, 0.85, "Misinformation", "TF-IDF + LogReg")
box(8.1, 0.55, 2.2, 0.85, "Alerts", "neg. spikes, flagged content", "#f9e0e0")
# sink
box(10.9, 3.6, 2.0, 1.0, "Elasticsearch", "socintel-posts", "#e9dff5")
box(10.9, 1.9, 2.0, 1.0, "Kibana", "search, trends, alerts", "#e9dff5")
box(10.9, 5.1, 2.0, 0.8, "Neo4j", "User graph", "#e9dff5")

arrow(2.2, 5.05, 3.0, 4.6)
arrow(2.2, 3.75, 3.0, 4.2)
arrow(4.9, 4.45, 5.4, 4.45)
for y in (5.62, 4.52, 3.42):
    arrow(7.5, 4.45, 8.1, y)
arrow(7.5, 4.2, 8.1, 2.5)             # clean text -> misinformation scoring
arrow(2.2, 2.3, 8.1, 2.2)             # LIAR -> classifier training
arrow(9.2, 1.9, 9.2, 1.4)             # scored posts -> alerts
arrow(10.3, 5.6, 10.9, 4.4)           # sentiment -> ES
arrow(10.3, 4.5, 10.9, 4.2)           # topics -> ES
arrow(10.3, 3.4, 10.9, 3.9)           # graph -> ES
arrow(10.3, 3.2, 10.9, 5.3)           # graph -> Neo4j
arrow(10.3, 2.4, 10.9, 3.7)           # misinfo -> ES
arrow(11.9, 3.6, 11.9, 2.9)           # ES -> Kibana
arrow(10.3, 0.95, 10.9, 2.0)          # alerts -> Kibana
ax.text(3.8, 0.9, "Each stage reads and writes data/processed/*.jsonl, so any stage can be rerun alone.\n"
        "Every document follows the shared schema in src/socintel/schema.py.",
        ha="center", fontsize=9, style="italic")
ax.set_title("Social Media Intelligence Platform - data flow", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=150)