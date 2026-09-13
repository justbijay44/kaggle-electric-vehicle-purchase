# EV Purchase Prediction

Predicts whether a person will buy an electric vehicle, based on the [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9) competition.

- **Model:** LightGBM, tuned, with leak-safe target encoding
- **Validation:** 5-fold stratified CV, pooled OOF ROC AUC ≈ **0.942**
- **Serving:** FastAPI backend + Streamlit frontend, containerized, deployed via CI/CD

## Architecture

```
train.py → model.pkl → FastAPI (api/) → Streamlit (app/)
```

The model is trained offline and saved as an artifact. The API loads it once and serves predictions over HTTP; Streamlit is a thin client that calls the API and displays the result. Each service runs in its own Docker container.

## Project structure

```
src/            # data loading, feature engineering, training, inference
api/            # FastAPI app (/predict endpoint)
app/            # Streamlit UI
tests/          # unit tests for src/
notebook/       # exploratory analysis and modeling notebooks
Dockerfile.api, Dockerfile.streamlit, docker-compose.yml
.github/workflows/
  tests.yml     # runs unit tests + validates Docker builds on every push
  deploy.yml    # trains the model, builds & pushes images to GHCR on push to main
```

## Running locally

```
pip install -r requirements.txt
python -m src.train          # downloads data, trains, saves models/model.pkl
uvicorn api.main:app --reload
streamlit run app/app.py
```

## Running with Docker

```
docker compose up --build
```

API on `localhost:8000`, Streamlit UI on `localhost:8501`.

## CI/CD

- Every push: unit tests run, Docker builds are validated
- Every push to `main`: the model is retrained from scratch (Kaggle credentials via GitHub secrets), and both images are rebuilt and pushed to GitHub Container Registry
