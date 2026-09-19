# 🛡️ SpamHam AI — Enterprise Spam & Phishing Detection Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An intelligent, state-of-the-art **Spam Detection & Phishing Analysis Platform** powered by Machine Learning and NLP. Features real-time inference, model explainability (suspicious keyword highlighting), calibrated probability confidence scoring, batch CSV dataset inspection, and high-performance REST APIs.

---

## 🌟 Key Features

- 🔮 **Interactive Web Studio**: Real-time message classification with preset test cases for instant evaluation.
- 🎯 **Calibrated Confidence Scoring**: Calculates exact Spam vs. Ham probability percentages with adaptive risk grading (*Low, Moderate, High Risk*).
- 🚩 **Explainable AI (XAI)**: Identifies and highlights suspicious keywords and spam triggers in real-time.
- 📊 **Text Characteristics Analytics**: Real-time extraction of uppercase ratio, URL count, word count, and character length.
- 📁 **Batch CSV Inspector**: Upload and classify full datasets in bulk with downloadable classification reports.
- 📈 **Live Model Performance Dashboard**: Displays Accuracy, Precision, Recall, F1 Score, and Confusion Matrix metrics.
- ⚡ **High-Throughput REST API**: Built on FastAPI with automatic interactive Swagger & ReDoc documentation.
- 🐳 **Production-Ready Deployment**: Containerized with Docker, health checks, and configurable environment settings.

---

## 🏗️ System Architecture & Directory Structure

```text
spamham/
├── app.py                     # FastAPI web server and REST API endpoints
├── train_and_export.py        # ML training & model serialization pipeline
├── upload_data_mongodb.py     # MongoDB dataset synchronizer
├── requirements.txt           # Python package dependencies
├── setup.py                   # Package setup configuration
├── Dockerfile                 # Production Docker container definition
├── .dockerignore              # Docker build exclusions
├── .gitignore                 # Git ignore rules for clean repo tracking
├── .env.example               # Template environment configuration
├── spamham.csv                # Dataset for model training
├── artifacts/                 # Pre-trained models and evaluation metrics
│   ├── best_model.pkl         # Trained TF-IDF + Classifier pipeline
│   └── model_metrics.json     # Performance scores & top feature weights
├── config/                    # Pipeline configuration files
│   ├── model.yaml
│   ├── schema.yaml
│   └── prediction_schema.yml
├── templates/                 # Frontend Jinja2 / HTML templates
├── static/                    # Static assets (CSS, JS, images)
├── notebooks/                 # Exploratory data analysis (EDA) notebooks
├── flowchart/                 # Architecture & pipeline diagrams
└── src/                       # Modular production pipeline package
    ├── components/            # Data ingestion, transformation, model trainer
    ├── configuration/         # Database and cloud storage connectors
    ├── constant/              # Global constants and schemas
    ├── entity/                # Data and config entities
    ├── ml/                    # Custom estimators and metric trackers
    └── pipeline/              # Training and prediction orchestrators
```

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- **Python 3.10+** (Python 3.10 - 3.13 supported)
- **Git**

### 2. Clone and Setup Environment

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (CMD):
# venv\Scripts\activate.bat
# On Windows (PowerShell):
# venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```
*(Optionally edit `.env` to configure MongoDB or AWS credentials if using cloud storage).*

### 5. Train & Export Model Pipeline (Optional)
The repository includes a pre-trained model in `artifacts/`. If you wish to retrain from `spamham.csv`:

```bash
python train_and_export.py
```

### 6. Run the Application

```bash
python app.py
```

Open your browser at:
- 🌐 **Web Studio Dashboard**: [http://localhost:8080](http://localhost:8080)
- ⚡ **Interactive Swagger API Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)
- 📖 **ReDoc Documentation**: [http://localhost:8080/redoc](http://localhost:8080/redoc)

---

## 🐳 Docker Deployment

### 1. Build Docker Image

```bash
docker build -t spamham-ai:latest .
```

### 2. Run Container

```bash
docker run -d \
  -p 8080:8080 \
  --name spamham-app \
  --restart unless-stopped \
  spamham-ai:latest
```

Visit `http://localhost:8080` to access the application.

---

## ☁️ Cloud Deployment Guides

### Option 1: Deploy to Render
1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your GitHub repository.
3. Configure the service:
   - **Environment**: `Docker` or `Python 3`
   - **Build Command** (if Python): `pip install -r requirements.txt`
   - **Start Command** (if Python): `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`

---

### Option 2: Deploy to Railway
1. Go to [Railway](https://railway.app) and create a new project.
2. Select **Deploy from GitHub repo**.
3. Railway will automatically detect the `Dockerfile` and build the container.
4. Set environment variable `PORT` to `8080` (or Railway's default).

---

### Option 3: Deploy to Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Docker** as the Space SDK.
3. Push this repository to your Space repository.

---

### Option 4: Deploy to AWS EC2 / DigitalOcean / Linux VPS
1. SSH into your server.
2. Install Docker and clone the repository.
3. Build and run using Docker:
   ```bash
   docker build -t spamham .
   docker run -d -p 80:8080 --restart always spamham
   ```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web Studio Dashboard interface |
| `GET` | `/health` | Healthcheck endpoint for load balancers & orchestrators |
| `POST` | `/api/v1/predict` | Single text prediction with explainability and metrics |
| `POST` | `/api/v1/predict-batch` | Batch list text classification |
| `POST` | `/api/v1/upload-csv` | CSV dataset classification |
| `GET` | `/api/v1/model-info` | Model metadata, evaluation scores & feature importances |
| `POST` | `/api/v1/train` | Asynchronously trigger model retraining |

---

### API Request & Response Examples

#### 1. Single Text Analysis (`POST /api/v1/predict`)

**Request:**
```bash
curl -X POST "http://localhost:8080/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "URGENT! You have won a 1000 Walmart Gift card. Claim here http://bit.ly/spam"}'
```

**Response:**
```json
{
  "is_spam": true,
  "label": "Spam",
  "prediction": 1,
  "spam_probability": 0.982,
  "confidence_percent": 98.2,
  "risk_level": "High Risk",
  "flagged_keywords": ["urgent", "won", "gift", "claim", "http"],
  "text_stats": {
    "char_count": 82,
    "word_count": 13,
    "uppercase_pct": 14.6,
    "url_count": 1
  }
}
```

---

#### 2. Model Performance Info (`GET /api/v1/model-info`)

**Response:**
```json
{
  "model_name": "LogisticRegression",
  "trained_at": "2025-04-13T12:00:00",
  "records_trained": 5572,
  "accuracy": 0.985,
  "precision": 0.991,
  "recall": 0.945,
  "f1_score": 0.967,
  "confusion_matrix": [[965, 1], [8, 141]],
  "top_spam_keywords": ["free", "txt", "claim", "call", "urgent", "prize", "won"]
}
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PORT` | `8080` | Port for the FastAPI server |
| `MONGODB_URL` | `mongodb://localhost:27017` | MongoDB connection string (Optional) |
| `MONGO_DATABASE_NAME` | `spamham_db` | Database name in MongoDB (Optional) |
| `MONGO_COLLECTION_NAME` | `spam_ham` | Collection name for dataset (Optional) |
| `AWS_ACCESS_KEY_ID` | `""` | AWS S3 Access Key (Optional for cloud storage) |
| `AWS_SECRET_ACCESS_KEY` | `""` | AWS S3 Secret Access Key (Optional) |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS S3 Region (Optional) |
| `MODEL_BUCKET_NAME` | `spamham-models` | AWS S3 Bucket Name for model artifacts (Optional) |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Abhishek Bisht**  
- Email: [abhiishekbishtt@gmail.com](mailto:abhiishekbishtt@gmail.com)
- GitHub: [@abhishekkbisht](https://github.com/abhishekkbisht)
