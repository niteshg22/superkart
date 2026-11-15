# SuperKart Sales Forecast: MLOps Pipeline

Automated pipeline that validates data, prepares train/test sets, tunes and tracks models with MLflow, commits the best model and serves it through a Streamlit app.

```
SuperKart_Project/
├── data/SuperKart.csv            # raw dataset
├── src/
│   ├── register_data.py          # 1. data registration and validation
│   ├── data_prep.py              # 2. cleaning, features, train/test split
│   └── train_model.py            # 3. tuning + MLflow tracking + best model
├── model/                        # best_model.joblib and metrics.json (committed by the pipeline)
├── app.py                        # Streamlit front end
├── requirements.txt
└── .github/workflows/pipeline.yml
```

## Run locally
```bash
pip install -r requirements.txt
python src/register_data.py
python src/data_prep.py
python src/train_model.py
streamlit run app.py
```

## Deploy
1. Push this folder to a public GitHub repository (branch `main`).
2. The GitHub Actions workflow runs on push and commits the trained model to `model/`.
3. On Streamlit Community Cloud choose the repository, branch `main` and main file `app.py`.
