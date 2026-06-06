# BotScan - Bot vs Human Account Detection System

AI-powered system that classifies social media accounts as **Human**, **Bot**, or **Suspicious** using deep learning and behavioral heuristics. Supports **Instagram**, **Twitter/X**, and **Facebook**.

Built with PyTorch, FastAPI, and Vue.js.

---
## Author

Satyam Kumar Thakur

AI-based Instagram Bot Detection System built using Machine Learning, Python, FastAPI, and Data Analysis.

---

## Architecture

```
                    Frontend (Vue.js)
                         |
                    FastAPI Server
                         |
        +----------------+----------------+
        |                |                |
   Instagram         Twitter/X        Facebook
   Providers         Providers        Providers
   (Apify/HTTP)      (Apify/HTTP)    (Apify/HTTP)
        |                |                |
        +--------> Feature Extractor (27 features)
                         |
                  BotDetectorNet (PyTorch)
                         |
                  Heuristic Override Layer
                         |
                  Prediction + Risk Score
```

---

## Features

- **3-class classification**: Human, Bot, Suspicious
- **27 behavioral features** extracted from profile metadata, engagement patterns, posting timing, and content analysis
- **Residual MLP** with Focal Loss for class-imbalanced training
- **Heuristic override layer** catches obvious bots the ML model misses (mass-follow, bio declarations, username patterns)
- **Multi-platform support** with pluggable provider architecture
- **Quota-saving fallback**: free HTTP scraper first, paid Apify only when needed
- **Real-time frontend** with confidence meter, risk badges, and behavioral breakdown

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys (Apify, RapidAPI, etc.)
# Use INSTAGRAM_PROVIDER=mock for testing without API keys
```

### 3. Start the server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the app

Go to **http://localhost:8000** in your browser.

API docs: **http://localhost:8000/docs**

### 5. (Optional) Expose via ngrok

```bash
ngrok http 8000
```

---

## Project Structure

```
.
├── app/
│   ├── main.py                    # FastAPI app, provider factory, lifespan
│   ├── config.py                  # Environment settings (Pydantic)
│   ├── dependencies.py            # Dependency injection
│   ├── api/v1/endpoints/
│   │   ├── analysis.py            # POST /analyze + heuristics
│   │   ├── history.py             # History & stats endpoints
│   │   └── health.py              # Health check
│   ├── services/
│   │   ├── predictor.py           # Model inference + confidence penalty
│   │   ├── feature_extractor.py   # 27-feature extraction pipeline
│   │   ├── history.py             # In-memory cache & history
│   │   ├── social_base.py         # Abstract provider interface
│   │   ├── instagram/             # Instagram: mock, http, apify, instaloader, rapidapi
│   │   ├── twitter/               # Twitter: mock, http, apify
│   │   └── facebook/              # Facebook: mock, http, apify
│   ├── schemas/
│   │   ├── analysis.py            # API request/response models
│   │   └── social.py              # Universal profile schema
│   └── core/
│       ├── exceptions.py          # Custom exceptions
│       └── logging.py             # Logging config
│
├── model/
│   ├── network.py                 # BotDetectorNet (Residual MLP)
│   ├── train.py                   # Training pipeline
│   └── artifacts/                 # Trained model, scaler, metadata
│
├── data/
│   ├── generate_data.py           # Instagram synthetic data generator
│   ├── generate_platform_data.py  # Twitter/Facebook/Snapchat synthetic data
│   └── prepare_real_data.py       # Real dataset preparation
│
├── frontend/
│   └── botscan.html               # Vue.js 3 single-page application
│
├── tests/                         # Unit & integration tests
├── .env.example                   # Environment template
└── requirements.txt
```

---

## Model

### Architecture: BotDetectorNet

```
Input (27 features)
  -> Stem: Linear(27->128) + BatchNorm + ReLU + Dropout(0.3)
  -> ResBlock 1: [Linear(128->128) + BN + ReLU + Dropout] x2 + Skip Connection
  -> ResBlock 2: same
  -> Neck: Linear(128->64) + BN + ReLU + Dropout(0.24)
  -> Head: Linear(64->3) -> [Human, Bot, Suspicious]
```

### Training

| Parameter | Value |
|-----------|-------|
| Loss | Focal Loss (gamma=2.0) + class weights |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Early stopping | patience=12 on validation macro-F1 |
| Preprocessing | log1p compression + RobustScaler |

### Results

```
Test Accuracy:  99.78%
Macro F1:       99.74%
Bot Precision:  99.9%
Bot Recall:     99.9%
```

---

## 27 Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | followers_count | Number of followers |
| 2 | following_count | Number of accounts followed |
| 3 | posts_count | Total number of posts |
| 4 | avg_likes_per_post | Average likes per post |
| 5 | avg_comments_per_post | Average comments per post |
| 6 | posts_per_day | Posting frequency |
| 7 | follower_following_ratio | Followers / following |
| 8 | engagement_rate | Likes / followers (log-normalized for large accounts) |
| 9 | profile_pic | Has profile picture (0/1) |
| 10 | bio_length | Biography character count |
| 11 | has_url_in_bio | Has external link (0/1) |
| 12 | is_verified | Platform verified (0/1) |
| 13 | is_private | Private account (0/1) |
| 14 | account_age_days | Days since account creation |
| 15 | username_digit_ratio | Digits / username length |
| 16 | username_length | Username character count |
| 17 | night_activity_ratio | Posts between 22:00-06:00 UTC |
| 18 | avg_caption_length | Average caption length |
| 19 | hashtags_per_post | Average hashtags per post |
| 20 | mentions_per_post | Average @mentions per post |
| 21 | story_frequency | Estimated stories per day |
| 22 | reels_ratio | Video posts / total posts |
| 23 | comment_reply_rate | Estimated reply rate |
| 24 | unique_commenters_ratio | Estimated commenter diversity |
| 25 | likes_comments_ratio | Likes / comments (detects bought engagement) |
| 26 | posting_regularity | 0 = fixed intervals (bot), 1 = irregular (human) |
| 27 | following_to_followers_ratio | Following / followers (detects mass-follow) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze` | Analyze an account |
| GET | `/api/v1/analyze/{platform}/{username}` | Get cached result |
| GET | `/api/v1/history` | Analysis history |
| GET | `/api/v1/history/export` | Export as CSV |
| DELETE | `/api/v1/history` | Clear history |
| GET | `/api/v1/stats` | Aggregated statistics |
| GET | `/api/v1/health` | Health check |
| GET | `/` | Frontend UI |
| GET | `/docs` | Swagger API documentation |

### Example

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"username": "natgeo", "platform": "instagram"}'
```

---

## Risk Classification

| Score Range | Level | Description |
|-------------|-------|-------------|
| 0.00 - 0.25 | Low | Likely human |
| 0.25 - 0.50 | Medium | Some unusual patterns |
| 0.50 - 0.75 | High | Strong bot indicators |
| 0.75 - 1.00 | Critical | Almost certainly a bot |

Formula: `risk_score = P(Bot) + 0.5 * P(Suspicious)`

---

## Provider Configuration

| Platform | Provider | What it fetches | API Key Required |
|----------|----------|----------------|-----------------|
| Instagram | mock | Fake data (dev) | No |
| Instagram | http | Profile metadata only | No |
| Instagram | instaloader | Profile + 12 posts | No |
| Instagram | apify | Full data | Yes (Apify) |
| Twitter | mock | Fake data (dev) | No |
| Twitter | http | Profile via GraphQL/Nitter | No |
| Twitter | apify | Profile + tweets | Yes (Apify) |
| Facebook | mock | Fake data (dev) | No |
| Facebook | http | Page metadata | No |
| Facebook | apify | Posts + reactions | Yes (Apify) |

---

## Training Your Own Model

```bash
# Generate synthetic data
python data/generate_data.py
python data/generate_platform_data.py

# Train (multiplatform mode)
python model/train.py --mode multiplatform --epochs 80 --batch 256

# Artifacts saved to model/artifacts/
```

---

## License

This project is part of a Master's degree in AI/ML — Major Project, 2026.
