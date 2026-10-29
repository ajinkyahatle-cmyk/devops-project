# Insurance Premium Predictor: PyCaret + MLflow + CI/CD to Streamlit Cloud

Predicts an insurance charge from age, sex, BMI, children, smoker status and region.

```
insurance/
  schema.py     input validation (no PyCaret)
  train.py      setup -> compare_models -> tune_model -> automl -> finalize -> save
  metrics.py    metrics.json reader and the quality gate
  predict.py    loads the saved pipeline and predicts
  release.py    CD: builds the deployable tree, verifies the live app
app.py          Streamlit UI (no training logic)
scripts/publish_release.sh   pushes the release tree to the streamlit-release branch
tests/          unit tests + real (fast) PyCaret training + release-tree test
Dockerfile      portable container image (optional; Streamlit Cloud does not use it)
.github/workflows/
  ci.yml        lint -> test -> train -> quality gate -> image build -> deploy
  deploy.yml    CD: publish release branch -> verify live app
  rollback.yml  redeploy an earlier release
```

## Run locally (Python 3.9-3.11)
```bash
make install
make test
make train          # compare + tune 8 models, ~1.5 min; logs to ./mlruns
make mlflow         # http://localhost:5000 -> compare runs
make run            # http://localhost:8501
```
`make train-fast` trains 2 candidates without tracking (about 30 s).

## How CD works here
Streamlit Community Cloud redeploys an app whenever the branch it watches changes.
CI therefore does not push to the cloud directly. After every check passes on `main` it:

1. builds a runtime-only tree: `app.py`, the 4 runtime modules, the trained model, `static/version.json`;
2. pushes it as a new commit on the **`streamlit-release`** branch (history = deployment history);
3. Community Cloud notices and redeploys;
4. CI polls `<app-url>/app/static/version.json` until the live app reports the commit just shipped,
   and fails the run if it never does.

Training code, tests and data never reach production, and the model that is served is exactly the one that
passed the quality gate (hold-out R2 >= 0.80).

## One-time setup
1. Create a **public** GitHub repo and push this project to `main` (Streamlit's free tier is simplest with public repos).
2. Wait for the CI run to finish. The deploy job creates the `streamlit-release` branch.
   (If the push is refused: repo Settings -> Actions -> General -> Workflow permissions -> Read and write.)
3. Go to https://share.streamlit.io -> **Create app** -> choose your repo, branch **`streamlit-release`**, main file **`app.py`**.
4. **Before clicking Deploy:** Advanced settings -> Python version **3.11**. PyCaret 3.3 does not support
   newer Python, and the version cannot be changed later without deleting and recreating the app.
5. After it deploys, copy the app URL, then GitHub -> Settings -> Secrets and variables -> Actions -> Variables ->
   new variable `APP_URL` = `https://<your-app>.streamlit.app`. From now on every deploy is verified.
6. Optional: Settings -> Environments -> `production` -> **Required reviewers** turns Continuous Deployment
   into Continuous Delivery (a human approves each release).

## Rollback
Actions -> **rollback** -> Run workflow -> paste a commit SHA from the `streamlit-release` branch
(each is titled `Release <source sha>`). No retraining or rebuilding.

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| Verify step says "Could not read version.json" | The app is asleep or crashed. Open the URL (click wake-up if shown), check Manage app -> logs, re-run the job. |
| Verify step says "Stale release" | Community Cloud has not picked up the push. Manage app -> **Reboot app**. |
| App crashes at startup | Python version is not 3.11 (see step 4), or the model file is missing from the release branch. |
| CI quality gate fails | Hold-out R2 fell below 0.80. Inspect `models/leaderboard.csv` in the CI artifact before lowering the threshold. |
