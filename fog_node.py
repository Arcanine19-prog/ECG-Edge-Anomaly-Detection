from flask import Flask, request, jsonify
from collections import deque
from datetime import datetime
import threading

fog_app = Flask(__name__)

session_store = {}
alert_log = deque(maxlen=200)
store_lock = threading.Lock()


def assess_health(total, anomalies, avg_loss, max_loss):
    if total == 0:
        return {
            "verdict":    "Unknown",
            "risk_level": "none",
            "color":      "gray",
            "message":    "No beats analyzed yet.",
            "advice":     "Submit at least one prediction to assess health status."
        }

    rate = (anomalies / total) * 100

    if total < 3:
        color = "green" if anomalies == 0 else "amber"
        return {
            "verdict":    "Normal" if anomalies == 0 else "Anomaly detected",
            "risk_level": "none" if anomalies == 0 else "moderate",
            "color":      color,
            "message":    f"Only {total} beat(s) analyzed. Submit more for an accurate assessment.",
            "advice":     "Analyze at least 5 to 10 beats for a reliable health verdict."
        }

    if rate == 0:
        return {
            "verdict":    "Healthy",
            "risk_level": "none",
            "color":      "green",
            "message":    f"All {total} beats analyzed are normal.",
            "advice":     "No anomalies detected. Heart rhythm appears regular and healthy."
        }

    if 0 < rate <= 10:
        return {
            "verdict":    "Mostly Normal",
            "risk_level": "low",
            "color":      "teal",
            "message":    f"{anomalies} out of {total} beats flagged ({rate:.1f}% anomaly rate).",
            "advice":     "Minor irregularities detected. Likely benign. Continue monitoring."
        }

    if 10 < rate <= 25:
        return {
            "verdict":    "Mild Concern",
            "risk_level": "moderate",
            "color":      "amber",
            "message":    f"{anomalies} out of {total} beats flagged ({rate:.1f}% anomaly rate).",
            "advice":     "Moderate irregularity detected. Consider consulting a physician."
        }

    if 25 < rate <= 50:
        return {
            "verdict":    "Concerning",
            "risk_level": "high",
            "color":      "orange",
            "message":    f"{anomalies} out of {total} beats flagged ({rate:.1f}% anomaly rate).",
            "advice":     "Significant arrhythmia pattern. Medical consultation recommended."
        }

    return {
        "verdict":    "Critical",
        "risk_level": "critical",
        "color":      "red",
        "message":    f"{anomalies} out of {total} beats flagged ({rate:.1f}% anomaly rate).",
        "advice":     "Severe cardiac irregularity detected. Seek immediate medical attention."
    }


@fog_app.route("/fog/ingest", methods=["POST"])
def ingest():
    data       = request.json
    session_id = data.get("session_id", "default")
    is_anomaly = data.get("is_anomaly", False)
    loss_value = data.get("loss_value", 0.0)
    beat_index = data.get("beat_index", 0)

    with store_lock:
        if session_id not in session_store:
            session_store[session_id] = {
                "total":      0,
                "anomalies":  0,
                "losses":     [],
                "started_at": datetime.now().isoformat()
            }
        s = session_store[session_id]
        s["total"] += 1
        s["losses"].append(round(loss_value, 6))
        if is_anomaly:
            s["anomalies"] += 1
            alert_log.append({
                "session": session_id,
                "beat":    beat_index,
                "loss":    round(loss_value, 6),
                "time":    datetime.now().isoformat()
            })

    return jsonify({"status": "ok"})


@fog_app.route("/fog/summary", methods=["GET"])
def summary():
    with store_lock:
        sessions = {}
        for sid, s in session_store.items():
            total    = s["total"]
            anomalies = s["anomalies"]
            losses   = s["losses"]
            avg_loss = round(sum(losses) / len(losses), 6) if losses else 0
            max_loss = round(max(losses), 6) if losses else 0
            health   = assess_health(total, anomalies, avg_loss, max_loss)
            sessions[sid] = {
                "total_beats":   total,
                "anomalies":     anomalies,
                "anomaly_rate":  round((anomalies / total * 100), 2) if total else 0,
                "avg_loss":      avg_loss,
                "max_loss":      max_loss,
                "started_at":    s["started_at"],
                "health_status": health
            }
        return jsonify({
            "sessions":       sessions,
            "recent_alerts":  list(alert_log)[-20:],
            "total_sessions": len(sessions)
        })


@fog_app.route("/fog/clear", methods=["POST"])
def clear():
    with store_lock:
        session_store.clear()
        alert_log.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    print("Fog node starting on http://127.0.0.1:5001")
    fog_app.run(port=5001, debug=False)