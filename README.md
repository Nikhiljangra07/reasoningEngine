# LoRa Deep Reasoning Engine

A domain-specialist reasoning engine built to compete with frontier models (Claude Opus 4.6, OpenAI o3) in **one** domain: reasoning about human problems — career, relationships, business, life decisions.

The thesis: a domain-specialist system powered by Sonnet + purpose-built architecture beats general-purpose frontier models in its own lane.

## Architecture

5 fused domains organized as a Taoist Wu Xing (Five Elements) dual-cycle engine:

| Element | Domain | Role |
|---------|--------|------|
| Earth | Physics | Ground of reality. What IS happening mechanically. |
| Metal | Mathematics | Precision grid. Structures, measures, cuts noise. |
| Water | Psychology | Hidden depths. Why the human distorts variables. |
| Wood | Philosophy | Expansion. Questions the question itself. |
| Fire | Chemistry | Transformation/Governance. Decides what bonds. |

**63 concepts** across the 5 domains. Two cycles run simultaneously:
- **Sheng** (Generating): Philosophy → Chemistry → Physics → Maths → Psychology
- **Ke** (Controlling): each domain is challenged by a different domain than the one feeding it

Convergence happens when the constructive cycle's output survives the deconstructive cycle's challenge.

See [CLAUDE.md](CLAUDE.md) for the full architecture and decision log.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY

# 3. Run the server
python server.py
```

Then open **http://localhost:8100** in your browser.

## Project Structure

```
reasoningEngine/
├── server.py              # FastAPI server + UI host
├── run.py                 # Interactive CLI mode
├── requirements.txt
├── .env.example
├── Dockerfile
├── web/
│   └── index.html         # Chat-based UI
├── src/
│   ├── core/types.py      # Shared data types + framework IDs
│   ├── domains/           # 5 isolated domain islands
│   │   ├── physics/
│   │   ├── psychology/
│   │   ├── philosophy/
│   │   └── chemistry/
│   ├── maths/             # 9 internal math layers
│   ├── formation/         # Wu Xing orchestration, funnel, cache
│   └── llm/               # Async engine, prompts, speech, client
└── tests/
    └── test_integration.py
```

## API

### `POST /api/trace`

Run the full reasoning engine on a problem.

**Request:**
```json
{
  "question": "Describe your situation here...",
  "max_iterations": 2,
  "phase1_summary": ""
}
```

**Response:**
```json
{
  "speech": "LoRa's narrated response...",
  "trajectories": [...],
  "domains": {...},
  "ke": [...],
  "convergence": [...],
  "funnel": [...],
  "stats": {
    "calls": 22,
    "tokens": 46193,
    "cost": 0.34,
    "iterations": 2,
    "converged": false
  }
}
```

### `GET /health`

Returns `{"status": "ok"}` for liveness probes.

## Testing

```bash
PYTHONPATH=. python -m unittest tests.test_integration -v
```

## Deployment

The included `Dockerfile` builds a runnable container:

```bash
docker build -t lora-reasoning-engine .
docker run -p 8100:8100 -e ANTHROPIC_API_KEY=sk-ant-... lora-reasoning-engine
```

The server reads `PORT`, `HOST`, and `CORS_ORIGINS` from the environment.

## License

Proprietary. All rights reserved.
