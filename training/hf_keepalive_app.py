"""
SalesPath — HF Spaces Keepalive App

Serves a simple FastAPI app after training completes to keep
the HF Space alive and display training results.
"""
import os
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="SalesPath — Training Complete")

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/salespath_out"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "SalesPath Training"}


@app.get("/")
async def root():
    """Display training results page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SalesPath Training Complete</title>
        <meta charset="utf-8">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   max-width: 800px; margin: 40px auto; padding: 20px;
                   background: #0f172a; color: #e2e8f0; }
            h1 { color: #38bdf8; }
            h2 { color: #94a3b8; margin-top: 30px; }
            .card { background: #1e293b; border-radius: 12px; padding: 20px; margin: 16px 0; }
            .metric { display: inline-block; margin: 8px 16px 8px 0; }
            .metric .value { font-size: 24px; font-weight: bold; color: #4ade80; }
            .metric .label { font-size: 12px; color: #64748b; text-transform: uppercase; }
            img { max-width: 100%; border-radius: 8px; margin: 16px 0; }
            .badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
                     font-size: 12px; font-weight: bold; }
            .badge.success { background: #166534; color: #4ade80; }
            .badge.info { background: #1e3a5f; color: #38bdf8; }
            pre { background: #0f172a; padding: 12px; border-radius: 8px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>🏆 SalesPath Training Complete</h1>
        <p>Trained model has been uploaded to Hugging Face Hub.</p>
    """

    # Load eval results
    eval_path = OUTPUT_DIR / "eval_results.json"
    if eval_path.exists():
        try:
            eval_data = json.loads(eval_path.read_text())
            html += '<div class="card"><h2>Evaluation Results</h2>'
            for key, value in eval_data.items():
                if isinstance(value, (int, float)):
                    html += f'<div class="metric"><div class="value">{value:.3f}</div><div class="label">{key}</div></div>'
                else:
                    html += f'<pre>{json.dumps(value, indent=2)}</pre>'
            html += "</div>"
        except Exception:
            pass

    # Show reward graph
    graph_path = OUTPUT_DIR / "reward_graph.png"
    if graph_path.exists():
        import base64
        img_b64 = base64.b64encode(graph_path.read_bytes()).decode()
        html += f'<div class="card"><h2>Reward Curve</h2><img src="data:image/png;base64,{img_b64}" alt="Reward Graph"/></div>'

    # Show reward history stats
    history_path = OUTPUT_DIR / "reward_history.txt"
    if history_path.exists():
        lines = history_path.read_text().strip().splitlines()
        rewards = [float(line.split("\t")[-1]) for line in lines if line.strip()]
        if rewards:
            html += f"""
            <div class="card">
                <h2>Training Stats</h2>
                <div class="metric"><div class="value">{len(rewards)}</div><div class="label">Episodes</div></div>
                <div class="metric"><div class="value">{sum(rewards)/len(rewards):.4f}</div><div class="label">Mean Reward</div></div>
                <div class="metric"><div class="value">{max(rewards):.4f}</div><div class="label">Max Reward</div></div>
                <div class="metric"><div class="value">{min(rewards):.4f}</div><div class="label">Min Reward</div></div>
            </div>
            """

    html += """
        <div class="card">
            <h2>Next Steps</h2>
            <p>1. View model on Hugging Face Hub</p>
            <p>2. Run inference with the trained model</p>
            <p>3. Stop this Space to avoid billing</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/api/results")
async def api_results():
    """Return training results as JSON."""
    results = {}

    eval_path = OUTPUT_DIR / "eval_results.json"
    if eval_path.exists():
        try:
            results["eval"] = json.loads(eval_path.read_text())
        except Exception:
            pass

    history_path = OUTPUT_DIR / "reward_history.txt"
    if history_path.exists():
        lines = history_path.read_text().strip().splitlines()
        rewards = [float(line.split("\t")[-1]) for line in lines if line.strip()]
        if rewards:
            results["training"] = {
                "episodes": len(rewards),
                "mean_reward": sum(rewards) / len(rewards),
                "max_reward": max(rewards),
                "min_reward": min(rewards),
                "std_reward": __import__("statistics").stdev(rewards) if len(rewards) > 1 else 0,
            }

    return JSONResponse(results)
