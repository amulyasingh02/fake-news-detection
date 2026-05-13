# 🧠 Fake News Detection Framework

A real-time, multimodal fake news detection system that combines **NLP-based semantic analysis** with **metadata-driven classification** to identify misinformation from social media streams (Reddit) in real-time.

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for detecting fake news by leveraging:
- **BERT** - Fine-tuned transformer model for text-based fake news classification
- **Random Forest** - Metadata model analyzing engagement patterns and source credibility
- **Streamlit** - Interactive dashboard for real-time monitoring and visualization
- **Reddit API** - Real-time data streaming from multiple subreddits

The system processes continuous Reddit streams, scores posts using a hybrid approach, and visualizes suspicious content through an interactive dashboard.

## 🏗️ Architecture

```
Reddit API / JSON Endpoint
        ↓
   fetch_data.py
        ↓
reddit_stream.csv
        ↓
Multimodal Prediction Engine
   ├── BERT Fake News Model
   └── Metadata Model (Random Forest)
        ↓
Score Fusion Layer
        ↓
Fake News Probability Score
        ↓
Streamlit Dashboard
```

## 📁 Project Structure

```
fake-news-detection/
├── pipeline/                          # Core ML pipeline
│   ├── fetch_data.py                 # Real-time Reddit data fetcher
│   ├── multimodal_predict.py         # Prediction engine with score fusion
│   └── train_metadata_model.py       # Random Forest model training
├── dashboard/                         # Interactive visualization
│   ├── app.py                        # Alternative dashboard (app.py)
│   └── app1.py                       # Main Streamlit dashboard
├── models/                           # Trained models
│   └── metadata_model.pkl            # Pre-trained Random Forest model
├── utils/                            # Utility modules
│   ├── labels.py                     # Dataset labeling utility
│   └── schema.py                     # Data preprocessing & schema
├── data/                             # Data storage
│   ├── reddit_stream.csv             # Raw streaming data
│   └── scored_reddit_posts.csv       # Predictions output
├── run.ps1                           # PowerShell execution script
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Conda or pip
- Internet connection (for Reddit API access)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/amulyasingh02/fake-news-detection.git
cd fake-news-detection
```

2. Create a conda environment:
```bash
conda create -n fake-news-env python=3.10
conda activate fake-news-env
```

3. Install required dependencies:
```bash
pip install requests pandas joblib transformers streamlit streamlit-option-menu streamlit-autorefresh plotly scikit-learn
```

### Configuration

Update file paths in the following scripts to match your system:
- `pipeline/fetch_data.py` - Set `DATA_PATH` 
- `pipeline/multimodal_predict.py` - Set `INPUT_PATH`, `OUTPUT_PATH`, `MODEL_PATH`
- `dashboard/app1.py` - Set `DATA_PATH`

## 🔄 Pipeline Components

### 1. **Data Fetcher** (`pipeline/fetch_data.py`)
- Monitors 5 subreddits: news, worldnews, politics, technology, science
- Fetches new posts every 120 seconds
- Extracts metadata: upvotes, comments, awards, user karma, etc.
- Prevents duplicate entries with post ID tracking
- Appends new data to `reddit_stream.csv`

**Key Features:**
- Continuous streaming with configurable intervals
- Reddit API integration with user-agent headers
- Duplicate detection using post IDs
- Timestamp recording for temporal analysis

### 2. **Prediction Engine** (`pipeline/multimodal_predict.py`)
- **Text Analysis**: BERT model (mrm8488/bert-tiny-finetuned-fake-news-detection)
- **Metadata Analysis**: Trained Random Forest classifier
- **Score Fusion**: Weighted combination (70% text, 30% metadata)
- Real-time processing with 5-second polling intervals

**Scoring Logic:**
```
fake_probability = 0.7 × text_score + 0.3 × metadata_score
prediction = fake_probability > 0.3 (binary classification)
```

**Metadata Features:**
- # upvotes, # comments, # awards
- Community members, years of membership
- Post karma, comment karma

### 3. **Model Training** (`pipeline/train_metadata_model.py`)
- Trains Random Forest model on labeled metadata features
- Saves model as `metadata_model.pkl`
- Can be retrained on updated datasets

### 4. **Dashboard** (`dashboard/app1.py`)
Interactive Streamlit interface with 4 pages:

#### Dashboard Page
- Real-time metrics: Total posts, fake predictions, average risk score, high-risk count
- Risk distribution histogram (fake_probability)
- Engagement vs. risk scatter plot (upvotes vs. fake_probability)

#### Suspicious Posts Page
- Top 20 highest-risk posts sorted by fake_probability
- Color-coded risk levels: HIGH (>0.8), MODERATE (0.6-0.8), LOW (<0.6)
- Visual progress bars and engagement metrics

#### Analytics Page
- Text vs. Metadata score correlation scatter plot
- Fake/Real prediction distribution pie chart
- Top 10 viral posts table with engagement metrics

#### Working Page
- System architecture documentation
- Pipeline flow visualization
- Component descriptions

**Features:**
- Auto-refresh every 5 seconds
- Responsive Plotly visualizations
- Real-time timestamp updates
- Mobile-friendly layout

## 📊 Data Schema

**Input Data** (`reddit_stream.csv`):
| Column | Type | Description |
|--------|------|-------------|
| post_id | str | Unique Reddit post identifier |
| Post text | str | Title + body text combined |
| # upvotes | int | Number of upvotes |
| # comments | int | Number of comments |
| # awards | int | Number of awards received |
| Community members | int | Subreddit subscriber count |
| Years of membership | float | Subreddit age in years |
| # Post Karma | int | Author's post karma |
| # Comment Karma | int | Author's comment karma |
| timestamp | datetime | UTC timestamp of fetch |

**Output Data** (`scored_reddit_posts.csv`):
Includes all input columns plus:
| Column | Type | Description |
|--------|------|-------------|
| text_score | float | BERT model confidence [0, 1] |
| meta_score | float | Random Forest confidence [0, 1] |
| fake_probability | float | Fused prediction score [0, 1] |
| prediction | int | Binary classification (0=real, 1=fake) |

## 🛠️ Running the System

### Option 1: Using PowerShell Script (Windows)
```powershell
.\run.ps1
```
This launches three concurrent processes:
1. Data fetcher
2. Prediction engine
3. Streamlit dashboard

### Option 2: Manual Execution
```bash
# Terminal 1: Start data fetcher
python pipeline/fetch_data.py

# Terminal 2: Start prediction engine
python pipeline/multimodal_predict.py

# Terminal 3: Launch dashboard
streamlit run dashboard/app1.py
```

The dashboard will be available at `http://localhost:8501`

## 🤖 Models Used

### BERT Text Classification
- **Model**: `mrm8488/bert-tiny-finetuned-fake-news-detection`
- **Input**: Post text (truncated to 512 tokens)
- **Output**: Fake news probability score
- **Framework**: Hugging Face Transformers

### Random Forest Metadata Classifier
- **Type**: Scikit-learn RandomForest
- **Features**: 7 metadata features
- **Serialization**: Joblib (metadata_model.pkl)
- **Output**: Credibility score based on engagement patterns

## 📈 Key Metrics

- **Detection Speed**: ~5 seconds per batch
- **Subreddit Coverage**: 5 major news/discussion communities
- **Real-time Latency**: <10 seconds from post to prediction
- **Data Retention**: Continuous streaming with no data loss
- **Scalability**: Easily expandable to more subreddits

## 🔧 Utilities

### `utils/labels.py`
- Processes labeled metadata datasets
- Identifies fake news communities
- Creates binary labels (0=real, 1=fake)
- Outputs: `metadata_labeled_dataset.csv`

### `utils/schema.py`
- Standardizes dataset schema
- Handles missing values (fills with 0)
- One-hot encodes categorical features (Post type, Flair, Community)
- Outputs: `metadata_training_dataset.csv`

## ⚙️ Configuration Parameters

**Tunable Parameters:**

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `SUBREDDITS` | fetch_data.py | 5 communities | Reddit communities to monitor |
| `FETCH_INTERVAL` | fetch_data.py | 120s | Data fetching interval |
| `text_weight` | multimodal_predict.py | 0.7 | BERT score weight in fusion |
| `meta_weight` | multimodal_predict.py | 0.3 | RF score weight in fusion |
| `fake_threshold` | multimodal_predict.py | 0.3 | Probability threshold for fake classification |
| `high_risk_threshold` | app1.py | 0.8 | Threshold for "HIGH RISK" label |
| `autorefresh_interval` | app1.py | 5000ms | Dashboard refresh interval |

## 📝 Usage Examples

### Adding New Subreddits
Edit `pipeline/fetch_data.py`:
```python
SUBREDDITS = [
    "news",
    "worldnews", 
    "politics",
    "technology",
    "science",
    "your_new_subreddit"  # Add here
]
```

### Adjusting Detection Sensitivity
Edit `pipeline/multimodal_predict.py`:
```python
# Make model more sensitive (lower threshold)
new_rows["prediction"] = (
    new_rows["fake_probability"] > 0.2  # Changed from 0.3
).astype(int)
```

### Retraining the Metadata Model
```bash
python pipeline/train_metadata_model.py
```
Ensure `metadata_training_dataset.csv` exists with labeled data.

## 📦 Dependencies

```
requests==2.31.0          # Reddit API calls
pandas==2.0.0             # Data manipulation
joblib==1.3.0             # Model serialization
transformers==4.30.0      # BERT model
scikit-learn==1.2.0       # Random Forest
streamlit==1.28.0         # Dashboard framework
streamlit-option-menu==0.3.2
streamlit-autorefresh==0.0.1
plotly==5.17.0            # Interactive visualizations
```

## 🚀 Future Enhancements

- [ ] Multi-language support beyond English
- [ ] Image/video content analysis
- [ ] Additional social media platforms (Twitter, Facebook)
- [ ] Ensemble methods combining multiple BERT variants
- [ ] Database integration for long-term storage
- [ ] Real-time alert notifications
- [ ] API endpoint for external integrations
- [ ] Model explainability (SHAP values)
- [ ] A/B testing framework for model improvements

## ⚠️ Limitations & Considerations

1. **API Rate Limiting**: Reddit API has rate limits; adjust `FETCH_INTERVAL` accordingly
2. **Model Accuracy**: BERT-tiny is lightweight but may trade accuracy for speed
3. **Cold Start**: Initial runs may process historical data
4. **Local Paths**: Script paths are hardcoded; update for your system
5. **Real-time Performance**: Dashboard responsiveness depends on system resources

## 📄 License

This project is provided as-is for educational and research purposes.

## 👨‍💻 Author

**Amulya Singh**  
GitHub: [@amulyasingh02](https://github.com/amulyasingh02)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## 📞 Support

For questions or issues, please open a GitHub issue in the repository.

---

**Last Updated**: May 2026  
**Status**: Active Development
