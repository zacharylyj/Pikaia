"""
generate_arch.py  —  Pikaia architecture map.
Run:  python generate_arch.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── Palette ──────────────────────────────────────────────────────────────────
BG         = "#0f1117"
C_ORCH     = "#5c6bc0"
C_AGENT    = "#3d4f9f"
C_MEM      = "#2e7d6f"
C_FILE     = "#1565c0"
C_PROV     = "#6a1b9a"
C_DS       = "#e65100"
C_TOOLS    = "#1b5e20"
C_OBS      = "#37474f"
C_HUMAN    = "#b71c1c"
C_IO       = "#212121"
T_LIGHT    = "#e8e8e8"
T_DIM      = "#9e9e9e"
T_TINY     = "#707070"
A_PRIM     = "#7986cb"
A_MEM      = "#4db6ac"
A_FEED     = "#ef9a9a"
A_DS       = "#ffb300"
A_FILE     = "#4fc3f7"
A_OBS      = "#78909c"

FW, FH = 24, 36
fig, ax = plt.subplots(figsize=(FW, FH))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")


# ── Primitives ────────────────────────────────────────────────────────────────
def box(x, y, w, h, title, sub="", color=C_ORCH, fs=10, sfs=8,
        r=0.35, zo=3, alpha_hex="cc"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={r}",
                          lw=1.4, edgecolor=color,
                          facecolor=color + alpha_hex, zorder=zo)
    ax.add_patch(rect)
    dy = 0.20 if sub else 0.0
    ax.text(x + w/2, y + h/2 + dy, title,
            ha="center", va="center", color=T_LIGHT,
            fontsize=fs, fontweight="bold", zorder=zo+1)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.22, sub,
                ha="center", va="center", color=T_DIM,
                fontsize=sfs, zorder=zo+1, style="italic")


def arr(x1, y1, x2, y2, color=A_PRIM, lw=1.8, rad=0.0, zo=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=zo)


def alabel(x, y, text, color=T_DIM, fs=7.2):
    ax.text(x, y, text, color=color, fontsize=fs,
            va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=BG+"bb",
                      edgecolor="none"))


def sechead(x, y, text, color):
    ax.text(x, y, text, color=color, fontsize=8, fontweight="bold",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                      edgecolor=color, lw=0.8))


# ═══════════════════════════════════════════════════════════════════════════════
# Column geometry
# ═══════════════════════════════════════════════════════════════════════════════
LX, LW = 0.5,  5.6     # left  – memory
CX, CW = 7.2,  7.2     # centre – spine
RX, RW = 15.6, 5.6     # right  – providers + file
BH = 1.05              # standard box height

# Row y-positions (bottom of each box) — generous 2.8-unit pitch
R = {}
R["title"]    = 35.0
R["input"]    = 33.2
R["main"]     = 30.8
R["orch"]     = 28.2
R["skill"]    = 25.7
R["dispatch"] = 23.2
R["handshake"]= 20.8
R["agents"]   = 17.8   # taller: 2.6 h container
AGENT_H = 2.6
R["providers"]= 14.8
R["toolreg"]  = 12.2
# tool-group rows live between 10.0 and 12.2
R["obs"]      =  8.2
R["postproc"] =  5.6
R["output"]   =  3.2
R["legend"]   =  0.0


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FW/2, 35.7, "Pikaia  —  Architecture Map",
        ha="center", va="center", color=T_LIGHT, fontsize=20, fontweight="bold",
        path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
ax.text(FW/2, 35.15,
        "v2  ·  4 agent tiers  ·  26 tools  ·  MemPalace v2  ·  DeepSeek-R1 fallback  ·  SQLite observability",
        ha="center", va="center", color=T_DIM, fontsize=9.5)


# ── Centre spine ──────────────────────────────────────────────────────────────
cx = CX + CW/2   # centre x

box(CX, R["input"],     CW, BH,  "User message",
    color=C_IO, fs=13)

box(CX, R["main"],      CW, BH,  "main.py  —  CLI entry",
    "--debug  |  --groq  |  --ollama  |  --deepseek  |  --project  |  --instance",
    color=C_IO)

box(CX, R["orch"],      CW, BH,  "Orchestrator",
    "classify intent · pick skill · dispatch agent",
    color=C_ORCH)

box(CX, R["skill"],     CW, BH,  "Skill Picker",
    "embed intent · cosine rank · lowest tier · SkillSmith if no match",
    color=C_ORCH)

box(CX, R["dispatch"],  CW, BH,  "Dispatch",
    "task packet · CT flag · worker slot · ContextManager.assess()",
    color=C_ORCH)

box(CX, R["handshake"], CW, BH,  "Agent Handshake",
    "ack · restate · planned steps · confidence",
    color=C_ORCH)

# Agent tier container
AY = R["agents"]
box(CX, AY, CW, AGENT_H, "", color=C_AGENT, r=0.5, zo=1, alpha_hex="55")
ax.text(cx, AY + AGENT_H - 0.3, "Agent Execution",
        ha="center", va="top", color=T_LIGHT, fontsize=11, fontweight="bold", zorder=4)
TW = (CW - 0.6) / 4
for i, (t, s) in enumerate([
    ("Tier 1", "atomic"),
    ("Tier 2", "composite\nmulti-step"),
    ("Tier 3", "decompose\nsynthesize"),
    ("Tier 4", "council\n3 parallel"),
]):
    tx = CX + 0.3 + i * (TW + 0.1)
    box(tx, AY + 0.35, TW, 1.85, t, s,
        color=C_AGENT, fs=9, sfs=7.5, zo=5)

box(CX, R["providers"], CW, BH,  "Provider Adapters",
    "build_request · call · parse_response",
    color=C_PROV)

box(CX, R["toolreg"],   CW, BH,  "Tool Registry  (26 tools)",
    "dispatch → impl/*.py → ToolResult { success, data, error }",
    color=C_TOOLS)

# Tool-group sub-boxes (2 rows × 3 cols)
TGW = (CW - 0.6) / 3
TGH = 0.88
tool_groups = [
    ("File & Code",  "read · write · edit · patch\nshell_exec · code_exec · delete · move"),
    ("Search",       "grep · glob · list"),
    ("Web & HTTP",   "web_fetch · web_search\nhttp_request"),
    ("Memory",       "memory_read · memory_write\ncontext_fetch"),
    ("LLM & Skills", "llm_call · embed_text\nskill_read · skill_write"),
    ("Lifecycle",    "ct_close · todo_write\nquestion · send_msg · cli_out"),
]
for i, (gn, gt) in enumerate(tool_groups):
    col_i, row_i = i % 3, i // 3
    gx = CX + 0.3 + col_i * (TGW + 0.15)
    gy = R["toolreg"] - 0.15 - (row_i+1) * (TGH + 0.12)
    box(gx, gy, TGW, TGH, gn, gt, color=C_TOOLS, fs=7.5, sfs=6.5, r=0.25, zo=4)

box(CX, R["obs"],       CW, BH,  "Observability",
    "db.py  ·  metrics.py  ·  trajectory.py  →  pikaia.db (SQLite WAL)",
    color=C_OBS)

box(CX, R["postproc"],  CW, BH,  "Post-process",
    "ST compress · history append · MT judge",
    color=C_ORCH)

box(CX, R["output"],    CW, BH,  "Interface output",
    "cli_output  ·  webapp stream",
    color=C_IO)


# ── Left column — memory layers ───────────────────────────────────────────────
sechead(LX, R["orch"] + BH + 0.18, "MEMORY LAYERS", C_MEM)

for by, title, sub in [
    (R["orch"]      + 0.08, "LT memory",
     "Long-term facts · full inject · global"),
    (R["skill"]     + 0.08, "MT memory  (MemPalace v2)",
     "Wing · Room · Hall · Tunnel · Recency decay · Dedup · Prune"),
    (R["dispatch"]  + 0.08, "KG  (Knowledge Graph)",
     "Temporal triples · contradiction detect · subject_timeline"),
    (R["handshake"] + 0.08, "CT state",
     "Active task flags · per-project · SkillSmith gate"),
    (R["agents"]    + 1.1,  "ST + History",
     "Window per-instance · compress to summary"),
]:
    box(LX, by, LW, 0.88, title, sub, color=C_MEM, fs=9, sfs=7)

# Human review
sechead(LX, R["providers"] + BH + 0.18, "HUMAN REVIEW", C_HUMAN)
box(LX, R["providers"] + 0.08, LW, 0.88,
    "Human review",
    "CT flag · low-confidence · SkillSmith approval",
    color=C_HUMAN, fs=9, sfs=7)


# ── Right column — providers ──────────────────────────────────────────────────
sechead(RX, R["orch"] + BH + 0.18, "LLM PROVIDERS", C_PROV)

providers = [
    ("Anthropic",       "claude-sonnet / haiku / opus"),
    ("OpenAI",          "gpt-4o · gpt-4o-mini"),
    ("Groq",            "llama-3.3-70b-versatile"),
    ("Ollama",          "local · text-inject fallback"),
    ("Debug",           "mock · no API key · canned JSON"),
]
for i, (name, note) in enumerate(providers):
    by = R["orch"] + 0.08 - i * 1.05
    box(RX, by, RW, 0.88, name, note, color=C_PROV, fs=9, sfs=7)

# DeepSeek — highlighted amber
DS_Y = R["orch"] + 0.08 - 5 * 1.05 - 0.18
box(RX, DS_Y, RW, 1.05,
    "DeepSeek-R1 1.5B  (local)",
    "Ollama first · transformers fallback\nno API key · automatic recovery",
    color=C_DS, fs=9, sfs=7.5)

# File context
sechead(RX, R["agents"] + AGENT_H + 0.18, "FILE CONTEXT", C_FILE)
for i, (name, note) in enumerate([
    ("File index  (L1)",      "always injected · path + summary"),
    ("File summaries  (L2)",  "RAG top-k · semantic search"),
    ("File content  (L3)",    "on-demand · file_read"),
]):
    by = R["agents"] + AGENT_H - 0.08 - i * 1.05
    box(RX, by, RW, 0.88, name, note, color=C_FILE, fs=9, sfs=7)

# pikaia.db box (right of obs)
box(RX, R["obs"] + 0.08, RW, 0.88,
    "pikaia.db  (SQLite WAL)",
    "trajectories · tool_events · metrics",
    color=C_OBS, fs=9, sfs=7)


# ═══════════════════════════════════════════════════════════════════════════════
# Arrows
# ═══════════════════════════════════════════════════════════════════════════════

# Centre spine
spine_pairs = [
    (R["input"]     + BH,  R["main"]     ),
    (R["main"]      + BH,  R["orch"]     ),
    (R["orch"]      + BH,  R["skill"]    ),
    (R["skill"]     + BH,  R["dispatch"] ),
    (R["dispatch"]  + BH,  R["handshake"]),
    (R["handshake"] + BH,  AY + AGENT_H  ),
    (AY,                   R["providers"] + BH),
    (R["providers"] + BH,  R["toolreg"]  ),
    (R["toolreg"]   - 2.1, R["obs"]      + BH),  # skip tool groups vertically
    (R["obs"]       + BH,  R["postproc"] ),
    (R["postproc"]  + BH,  R["output"]   ),
]
for y_from, y_to in spine_pairs:
    arr(cx, y_from, cx, y_to, color=A_PRIM, lw=2.2)

# Feedback loop: post-process → orch
arr(CX + 0.5, R["postproc"] + BH/2,
    CX + 0.5, R["orch"]     + BH/2,
    color=A_FEED, lw=1.4, rad=-0.45)
alabel(CX - 1.0, (R["postproc"] + R["orch"]) / 2 + BH/2,
       "next turn", A_FEED)

# Memory inject (left → centre)
for my, cy2 in [
    (R["orch"]      + 0.52, R["orch"]      + 0.52),
    (R["skill"]     + 0.52, R["skill"]     + 0.52),
    (R["dispatch"]  + 0.52, R["dispatch"]  + 0.52),
    (R["handshake"] + 0.52, R["handshake"] + 0.52),
    (R["agents"]    + 1.55, R["agents"]    + 1.55),
]:
    arr(LX + LW, my, CX, cy2, color=A_MEM, lw=1.3)

# Human review → dispatch
arr(LX + LW, R["providers"] + 0.52,
    CX,        R["dispatch"]  + 0.52,
    color=A_FEED, lw=1.2, rad=0.12)
alabel(LX + LW + 0.2, (R["providers"] + R["dispatch"])/2,
       "CT approval", A_FEED)

# Providers → provider adapter box (right → centre)
PAD_CY = R["providers"] + BH/2
for i in range(5):
    py = R["orch"] + 0.08 - i * 1.05 + 0.44
    arr(RX, py, CX + CW, PAD_CY, color="#ce93d8", lw=1.0)

# DeepSeek → provider adapter (amber)
arr(RX, DS_Y + 0.52, CX + CW, PAD_CY, color=A_DS, lw=2.0)
alabel(RX - 0.1, DS_Y + 0.52, "fallback", A_DS)

# Agent → DeepSeek direct (_try_deepseek_fallback)
arr(CX + CW, AY + AGENT_H/2,
    RX + RW,  DS_Y + 0.52,
    color=A_DS, lw=1.6, rad=0.3)
alabel(RX + RW + 0.1, (AY + AGENT_H/2 + DS_Y + 0.52)/2,
       "_try_deepseek_fallback", A_DS)

# File context → agent (right → centre)
for i in range(3):
    fy = R["agents"] + AGENT_H - 0.08 - i * 1.05 + 0.44
    arr(RX, fy, CX + CW, AY + AGENT_H/2, color=A_FILE, lw=1.1)

# Obs → pikaia.db (right)
arr(CX + CW, R["obs"] + BH/2,
    RX,        R["obs"] + BH/2 + 0.08,
    color=A_OBS, lw=1.4)
alabel(CX + CW + 0.15, R["obs"] + BH/2 + 0.2, "flush", A_OBS)

# Tool groups connector (vertical dashed hint)
ax.plot([cx, cx], [R["toolreg"], R["toolreg"] - 2.1],
        color=C_TOOLS + "88", lw=1.2, linestyle="dashed", zorder=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Legend
# ═══════════════════════════════════════════════════════════════════════════════
LEG_X, LEG_Y, LEG_W, LEG_H = 0.4, 0.05, FW - 0.8, 2.9
box(LEG_X, LEG_Y, LEG_W, LEG_H, "", color="#1a1f2e", r=0.6, zo=1, alpha_hex="ff")
ax.text(LEG_X + 0.35, LEG_Y + LEG_H - 0.15, "Legend",
        color=T_DIM, fontsize=8.5, fontweight="bold", va="top", zorder=4)

color_items = [
    (C_ORCH,  "Orchestration logic"),
    (C_AGENT, "Agent runtime"),
    (C_MEM,   "Memory layers"),
    (C_FILE,  "File context"),
    (C_PROV,  "LLM providers"),
    (C_DS,    "DeepSeek fallback"),
    (C_TOOLS, "Tool registry"),
    (C_OBS,   "Observability"),
    (C_HUMAN, "Human gates"),
    (C_IO,    "I/O + CLI"),
]
NCOLS = 5
for idx, (c, label) in enumerate(color_items):
    ci, ri = idx % NCOLS, idx // NCOLS
    lx = LEG_X + 0.4 + ci * 4.6
    ly = LEG_Y + LEG_H - 0.58 - ri * 0.82
    rect = FancyBboxPatch((lx, ly - 0.22), 0.38, 0.38,
                          boxstyle="round,pad=0.04",
                          facecolor=c + "cc", edgecolor=c, lw=1, zorder=5)
    ax.add_patch(rect)
    ax.text(lx + 0.52, ly - 0.03, label,
            color=T_LIGHT, fontsize=8, va="center", zorder=6)

arrow_items = [
    (A_PRIM,  "Primary flow"),
    (A_MEM,   "Memory inject"),
    (A_FEED,  "Feedback / CT"),
    (A_DS,    "DeepSeek fallback"),
    (A_FILE,  "File context inject"),
    (A_OBS,   "Observability flush"),
]
for idx, (c, label) in enumerate(arrow_items):
    ci, ri = idx % 3, idx // 3
    lx = LEG_X + 0.4 + ci * 7.6
    ly = LEG_Y + 0.88 - ri * 0.52
    ax.annotate("", xy=(lx + 0.55, ly), xytext=(lx, ly),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8), zorder=5)
    ax.text(lx + 0.7, ly, label, color=T_DIM, fontsize=8, va="center", zorder=6)


# ── Save ──────────────────────────────────────────────────────────────────────
out = r"D:\Repos\Pikaia\.claude\worktrees\romantic-shannon\architecture_map.png"
plt.tight_layout(pad=0)
fig.savefig(out, dpi=160, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print(f"Saved -> {out}")
plt.close(fig)
