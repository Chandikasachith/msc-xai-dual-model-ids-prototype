# Deploy the Streamlit dashboard

Host the dual-model intrusion detection dashboard on **Streamlit Community Cloud** (free) or any container host that runs `streamlit run`.

## What gets deployed

| Include in git | Exclude (local only) |
|----------------|----------------------|
| `streamlit_dashboard.py`, `cicids2017_preprocess.py`, `xai_insight_language.py`, `llm_openai_advisory.py`, `teams_executive_summary.py` | `.streamlit/secrets.toml` |
| `SecondModel/*.pkl` (models, scalers, encoders) | Full `SecondModel/X_test_*.npy` (~180 MB; over GitHub 100 MB limit) |
| `SecondModel/*_deploy.npy` (small LIME/sample slices) | `SecondModel/cicids-2017/` raw dataset |
| `requirements.txt`, `.streamlit/config.toml` | `archive/`, `Presentation_Evidence/` (optional) |
| Sample CSVs if you want (`test_2018.csv`, etc.) | API keys and webhook URLs |

## 1. Prepare deploy artifacts

From the project root (after models are trained locally):

```powershell
python prepare_streamlit_deploy.py
```

This writes `SecondModel/*_deploy.npy` (~300 rows each). The dashboard prefers these files automatically.

## 2. Initialize git and push to GitHub

```powershell
cd "c:\Users\UPEIRCH\Documents\MSC\Research\Research Development"
git init
git add requirements.txt DEPLOY.md prepare_streamlit_deploy.py streamlit_dashboard.py
git add cicids2017_preprocess.py xai_insight_language.py llm_openai_advisory.py teams_executive_summary.py
git add .streamlit/config.toml .streamlit/secrets.toml.example
git add SecondModel/*.pkl SecondModel/*_deploy.npy
git add test_2018.csv Datasample_200.csv
git commit -m "Add Streamlit deploy packaging for dual-model dashboard"
```

Create a new **private** GitHub repository, then:

```powershell
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

Use a **private** repo if the project is thesis/research work.

## 3. Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **New app** → select your repository.
3. **Main file path:** `streamlit_dashboard.py`
4. **Advanced settings → Python version:** choose **3.12** (not 3.14).
   - Streamlit Cloud **ignores** `.python-version` and `runtime.txt`.
   - If the app was already deployed on 3.14, **delete the app** and redeploy — you cannot change Python on an existing deployment.
5. **Advanced settings → Secrets**, paste:

```toml
TEAMS_WEBHOOK_URL = "https://your-organization.webhook.office.com/webhookb2/..."

# Optional: AI analyst suggestions in the dashboard
OPENAI_API_KEY = "sk-..."
# OPENAI_MODEL = "gpt-4o"
```

5. Deploy. First boot may take a few minutes while dependencies install.

Your app URL will look like: `https://YOUR_APP.streamlit.app`

## 4. Local secrets (development)

Copy the example file and edit locally (never commit `secrets.toml`):

```powershell
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

## Security notes

- A public app lets anyone upload CSVs and can trigger **OpenAI** and **Teams** calls if secrets are set.
- For a public demo, omit `OPENAI_API_KEY` and `TEAMS_WEBHOOK_URL` from Streamlit Cloud secrets.
- Prefer a **private** GitHub repo and share the Streamlit link only with your supervisor/committee.

## Other hosts

Same entrypoint everywhere:

```bash
streamlit run streamlit_dashboard.py
```

Works on **Hugging Face Spaces** (Streamlit SDK), **Render**, **Railway**, or a university VM/Docker image. Set secrets as environment variables or mount `.streamlit/secrets.toml` on the server.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Error loading RF` / missing `.npy` | Run `python prepare_streamlit_deploy.py` and commit `*_deploy.npy` |
| Git push rejected (large file) | Ensure full `X_test_*.npy` are gitignored; only commit `*_deploy.npy` |
| `ModuleNotFoundError: joblib` | App built on Python 3.14 — delete app, redeploy with **Python 3.12** in Advanced settings |
| Health check / connection reset | Usually app crash on startup — fix Python version first; check logs for import errors |
| OpenAI / Teams not working on cloud | Add secrets in Streamlit Cloud settings (not in the repo) |
| App sleeps / slow cold start | Normal on free tier; click **Load Models** after the page opens |
