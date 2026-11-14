# CreditSense UI

React (Vite) + TypeScript + Tailwind + Recharts front end for the CreditSense scoring API (`creditsense/api.py`).
Seven screens: Applicant decision, Policy simulator, Batch scoring, Model health, Fairness and reliability,
Research evidence, About and limitations.

## Run

```bash
# terminal 1, repository root (first time: create a virtual environment and install the pinned packages)
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt          # Windows: .venv\Scripts\python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn creditsense.api:app              # Windows: .venv\Scripts\python -m uvicorn creditsense.api:app
                                                             # API on http://127.0.0.1:8000 (`make serve` does the same where make exists)

# terminal 2
cd ui
npm install
npm run dev                     # http://localhost:5173
```

The API address comes from `VITE_API_BASE_URL` (see `.env.example`); it defaults to `http://127.0.0.1:8000`.

## Where the numbers come from

Nothing on screen is typed in by hand.

- Live results come from the API (`src/api.ts`): scores, decisions, reasons, policy simulation, group rates,
  monitoring and the model card. Form limits are read from the API's own OpenAPI schema.
- Research evidence comes from the repository's experiment outputs. `scripts/sync-data.mjs` copies
  `phase2/outputs/*.json` and `*.png`, the Phase I results and `data/demo/*.csv` into `public/repo/`
  before `npm run dev` and `npm run build`. That folder is git-ignored; the repository files stay the only source.
- If a file or the service is missing, the screen says so. It does not fall back to a stored number.

Slider ranges, the 2% recalibration tolerance and the 95% confidence option are controls from the build brief,
not results.

## Test and build

```bash
npm test            # component tests against a mocked API
npm run build       # type-check, then build to dist/
```

The API side of the contract is tested in `tests/test_api_contract.py` (`make test` from the repository root).

## Deploy

Backend: the `Dockerfile` at the repository root (Hugging Face Spaces or Render). Front end: build with
`VITE_API_BASE_URL` set to the backend address and host `dist/` (for example on Vercel, root directory `ui`).
