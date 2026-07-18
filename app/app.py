"""UFC fight predictor — Flask serving layer.

Loads the two weekly-retrained bundles and the fighter snapshots once at
startup; `/api/predict` assembles a hypothetical matchup and quotes win
probability, display odds, and the finish-method distribution.

Run from the project root:  python -m app.app   (http://127.0.0.1:5001)
"""
import json
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request

from app.utils import (fmt_height, head_to_head, load_fights, load_snapshots,
                       matchup_row, pro_record)
from src.config import APP_HOST, APP_PORT
from src.utils import load_models, predict_fight

app = Flask(__name__)

SNAPS = load_snapshots()
FIGHTS = load_fights()
WIN_BUNDLE, FINISH_BUNDLE = load_models()
# dropdown order: women's ladder, men's ladder light->heavy, then the odd ones
_DIV_ORDER = ["Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
              "Women's Featherweight",
              "Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
              "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
              "Catch Weight", "Open Weight"]
DIVISIONS_UI = sorted(SNAPS["last_division"].dropna().unique(),
                      key=lambda d: (_DIV_ORDER.index(d) if d in _DIV_ORDER else len(_DIV_ORDER), d))

# structured rows for the client-side combobox (name/record/division/elo shown
# per option, so same-name fighters stay distinguishable)
FIGHTERS = sorted(
    ({"id": fid, "name": s["name"], "record": pro_record(s),
      "division": s["last_division"], "elo": round(float(s["elo"]))}
     for fid, s in SNAPS.iterrows()),
    key=lambda f: f["name"])


@app.route("/")
def index():
    return render_template(
        "index.html",
        fighters_json=json.dumps(FIGHTERS),
        divisions=DIVISIONS_UI,
        n_fighters=len(SNAPS),
        trained_through=WIN_BUNDLE.get("trained_through", "?"),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    try:
        snap_a = SNAPS.loc[data["fighter_a"]]
        snap_b = SNAPS.loc[data["fighter_b"]]
    except KeyError:
        return jsonify({"error": "unknown fighter id"}), 400
    if data["fighter_a"] == data["fighter_b"]:
        return jsonify({"error": "pick two different fighters"}), 400

    row = matchup_row(snap_a, snap_b, data.get("division", "Lightweight"),
                      data.get("title_fight", False))
    pred = predict_fight(row, WIN_BUNDLE, FINISH_BUNDLE)

    now = datetime.now().date()

    def card(s):
        age = round((now - s["dob"].date()).days / 365.25, 1) if pd.notna(s["dob"]) else None
        return {"name": s["name"],
                "record": pro_record(s),
                "ufc_record": f"{int(s['career_wins'])}-{int(s['career_losses'])}",
                "streak": int(s["career_win_streak"]),
                "elo": round(float(s["elo"])),
                "division": s["last_division"],
                "last_fight": str(s["last_fight_date"].date()),
                # tale-of-the-tape stats
                "age": age,
                "height": fmt_height(s["height_in"]),
                "height_in": (round(float(s["height_in"])) if pd.notna(s["height_in"]) else None),
                "reach": (round(float(s["reach_in"])) if pd.notna(s["reach_in"]) else None),
                "reach_est": bool(s.get("reach_imputed", False)),
                "sig_strikes": round(float(s["career_avg_sig_str_landed"]), 1),
                "takedowns": round(float(s["career_avg_takedowns_landed"]), 2),
                "finish_rate": round(float(s["career_finish_rate"]) * 100),
                "win_rate": round(float(s["career_win_rate"]) * 100)}

    h2h = head_to_head(FIGHTS, data["fighter_a"], data["fighter_b"])
    return jsonify({"fighter_a": card(snap_a), "fighter_b": card(snap_b),
                    "h2h": h2h, **pred})


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=False)
