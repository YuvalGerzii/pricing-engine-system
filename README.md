# 🏠 Pricing Engine System

An AI-driven pricing engine system that takes internal sales data and external competitor data to build intelligent pricing rules per region and property type. The system provides comprehensive ETL pipeline, machine learning model training, pricing rule export, and interactive user interfaces.

## 🎯 System Overview

The Pricing Engine System consists of four main components:

1. **ETL Pipeline** - Ingests, cleans, and normalizes historical sales and competitor data
2. **AI Model Training** - Uses regression and tree-based models for optimal price predictions
3. **Pricing Rule Exporter** - Generates human-readable pricing rules in JSON and Excel formats
4. **User Interfaces** - FastAPI service and Streamlit UI for interactive pricing analysis

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ETL Pipeline  │───▶│  Model Training  │───▶│  Rule Exporter  │
│                 │    │                  │    │                 │
│ • Data Ingestion│    │ • XGBoost        │    │ • JSON Export   │
│ • Data Cleaning │    │ • Random Forest  │    │ • Excel Export  │
│ • Normalization │    │ • Linear Reg.    │    │ • Summaries     │
│ • Data Merging  │    │ • Cross Val.     │    │ • Rule Lookup   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js UI    │◀───│   FastAPI        │◀───│  Trained Model  │
│                 │    │                  │    │                 │
│ • Dashboard     │    │ • Price Queries  │    │ • Predictions   │
│ • Status Monitor│    │ • Rule Queries   │    │ • Confidence    │
│ • Quick Actions │    │ • Export API     │    │ • Metadata      │
└─────────────────┘    │ • Model Status   │    └─────────────────┘
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Streamlit UI    │
                       │                  │
                       │ • Price Analysis │
                       │ • Interactive UI │
                       │ • Export Tools   │
                       │ • Visualizations │
                       └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ (for Next.js frontend)
- Python 3.8+ (for pricing engine)
- pip (Python package manager)

### 1. Install Dependencies

**Frontend (Next.js):**
```bash
npm install
```

**Backend (Python):**
```bash
cd pricing_engine
pip install -r requirements.txt
```

### 2. Run the Complete System

**Step 1: Start the ETL Pipeline and Train Model**
```bash
cd pricing_engine
python etl_pipeline.py
python model_training.py
```

**Step 2: Start the FastAPI Service**
```bash
python fastapi_app.py
# API will be available at http://localhost:8001
```

**Step 3: Start the Streamlit UI**
```bash
streamlit run pricing_ui.py
# UI will be available at http://localhost:8501
```

**Step 4: Start the Next.js Dashboard**
```bash
npm run dev
# Dashboard will be available at http://localhost:8000
```

## 📊 System Components

### ETL Pipeline (`etl_pipeline.py`)

Handles data ingestion and processing:

- **Data Loading**: Supports CSV and Excel files
- **Data Cleaning**: Handles missing values, standardizes formats
- **Data Normalization**: Converts units, standardizes city names
- **Data Merging**: Combines internal and competitor data
- **Sample Data Generation**: Creates test datasets for development

**Usage:**
```bash
python etl_pipeline.py
```

**Input Files:**
- `pricing_engine/data/internal_sales.csv`
- `pricing_engine/data/competitor_data.csv`

**Output:**
- `pricing_engine/data/processed_data.csv`

### AI Model Training (`model_training.py`)

Trains machine learning models for price prediction:

- **Supported Models**: XGBoost, Random Forest, Gradient Boosting, Linear Regression
- **Feature Engineering**: Handles categorical variables, scaling
- **Model Evaluation**: Cross-validation, confidence intervals
- **Model Persistence**: Saves trained models with metadata

**Usage:**
```bash
python model_training.py
```

**Features Used:**
- Property size (m²)
- Floor number
- City (encoded)
- Property type (encoded)
- Data source

**Output:**
- `pricing_engine/models/pricing_model.pkl`

### Pricing Rule Exporter (`pricing_rule_exporter.py`)

Generates and exports pricing rules:

- **Rule Generation**: Creates rules for different property segments
- **Export Formats**: JSON, Excel, human-readable summaries
- **Rule Lookup**: Finds best matching rules for specific criteria
- **Segmentation**: By city, property type, size range, floor range

**Usage:**
```bash
python pricing_rule_exporter.py
```

**Example Rule:**
```json
{
  "Tel Aviv, 100-120 m², Apartment, Floor 3-5": {
    "segment": {
      "city": "Tel Aviv",
      "property_type": "Apartment",
      "size_range": "100-120 m²",
      "floor_range": "3-5"
    },
    "pricing": {
      "recommended_price_per_m2": 37500,
      "price_range": {
        "min": 35000,
        "max": 40000
      },
      "confidence_interval": 2500,
      "currency": "₪"
    },
    "rule_formatted": "₪37,500/m²"
  }
}
```

### FastAPI Service (`fastapi_app.py`)

REST API for programmatic access:

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /price-query` - Get price prediction
- `POST /rule-query` - Find matching pricing rule
- `GET /model-status` - Model information
- `GET /feature-importance` - Feature importance
- `POST /export-rules` - Export pricing rules
- `GET /download/{filename}` - Download exported files
- `GET /rules/summary` - Rules summary
- `GET /cities` - Available cities
- `GET /property-types` - Available property types

**Usage:**
```bash
python fastapi_app.py
# API docs: http://localhost:8001/docs
```

**Example API Call:**
```bash
curl -X POST "http://localhost:8001/price-query" \
  -H "Content-Type: application/json" \
  -d '{
    "city": "Tel Aviv",
    "property_type": "Apartment",
    "size": 100,
    "floor": 3,
    "data_source": "internal"
  }'
```

### Streamlit UI (`pricing_ui.py`)

Interactive web interface:

**Features:**
- **Price Queries**: Interactive form for price predictions
- **Visualizations**: Price range charts and analysis
- **Export Tools**: Download pricing rules in various formats
- **History Tracking**: Keep track of previous queries
- **Model Status**: View model information and performance

**Usage:**
```bash
streamlit run pricing_ui.py
# UI: http://localhost:8501
```

### Next.js Dashboard (`src/app/page.tsx`)

System monitoring and management dashboard:

**Features:**
- **Service Status**: Monitor all system components
- **System Overview**: Key metrics and statistics
- **Quick Actions**: Common tasks and operations
- **Architecture View**: System component overview
- **Setup Instructions**: Getting started guide

**Usage:**
```bash
npm run dev
# Dashboard: http://localhost:8000
```

## 📁 Project Structure

```
├── pricing_engine/                 # Python backend
│   ├── requirements.txt           # Python dependencies
│   ├── etl_pipeline.py            # Data processing
│   ├── model_training.py          # ML model training
│   ├── pricing_rule_exporter.py   # Rule generation & export
│   ├── fastapi_app.py             # REST API service
│   ├── pricing_ui.py              # Streamlit interface
│   ├── data/                      # Data files
│   │   ├── internal_sales.csv     # Sample internal data
│   │   ├── competitor_data.csv    # Sample competitor data
│   │   └── processed_data.csv     # Processed data
│   ├── models/                    # Trained models
│   │   └── pricing_model.pkl      # Saved model
│   ├── exports/                   # Exported files
│   └── tests/                     # Unit tests
├── src/                           # Next.js frontend
│   ├── app/
│   │   ├── page.tsx              # Main dashboard
│   │   ├── layout.tsx            # App layout
│   │   └── globals.css           # Global styles
│   ├── components/ui/            # UI components
│   └── lib/                      # Utilities
├── package.json                  # Node.js dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file for configuration:

```env
# API Configuration
FASTAPI_HOST=localhost
FASTAPI_PORT=8001

# Model Configuration
MODEL_TYPE=xgboost
CONFIDENCE_LEVEL=0.95

# Data Configuration
DATA_PATH=pricing_engine/data
MODEL_PATH=pricing_engine/models
EXPORT_PATH=pricing_engine/exports
```

### Model Configuration

Modify model parameters in `model_training.py`:

```python
# XGBoost Configuration
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

## 📈 Sample Data

The system includes sample data for testing:

**Internal Sales Data:**
- 8 property records
- Cities: Tel Aviv, Jerusalem, Haifa
- Property types: Apartment, House
- Price range: ₪33,000 - ₪45,000/m²

**Competitor Data:**
- 8 competitor records
- Same cities and property types
- Competitor names: CompA, CompB
- Price range: ₪34,000 - ₪43,000/m²

## 🧪 Testing

Run individual components:

```bash
# Test ETL Pipeline
cd pricing_engine
python etl_pipeline.py

# Test Model Training
python model_training.py

# Test Rule Export
python pricing_rule_exporter.py

# Test API
python fastapi_app.py
# Visit http://localhost:8001/docs

# Test UI
streamlit run pricing_ui.py
# Visit http://localhost:8501
```

## 📊 Performance Metrics

The system tracks various performance metrics:

- **Model Accuracy**: R² score, RMSE, MAE
- **Confidence Intervals**: Statistical confidence ranges
- **Feature Importance**: Most influential factors
- **Cross-Validation**: Model generalization performance
- **Processing Speed**: ETL and prediction times

## 🔒 Security Considerations

- **Input Validation**: All user inputs are validated
- **Error Handling**: Comprehensive error handling throughout
- **CORS Configuration**: Properly configured for production
- **File Access**: Restricted file system access
- **API Rate Limiting**: Consider implementing for production

## 🚀 Deployment

### Production Deployment

1. **Containerization** (Docker):
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY pricing_engine/ .
RUN pip install -r requirements.txt
CMD ["python", "fastapi_app.py"]
```

2. **Environment Setup**:
- Use environment variables for configuration
- Set up proper logging
- Configure database connections (if needed)
- Set up monitoring and alerting

3. **Scaling Considerations**:
- Use load balancers for API endpoints
- Implement caching for frequent queries
- Consider database for large datasets
- Set up automated model retraining

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:

1. Check the API documentation at `/docs`
2. Review the system logs
3. Test individual components
4. Check service status at `/health`

## 🔄 Updates and Maintenance

### Regular Maintenance Tasks:

1. **Model Retraining**: Retrain models with new data
2. **Data Updates**: Update sample data and schemas
3. **Dependency Updates**: Keep packages up to date
4. **Performance Monitoring**: Track system performance
5. **Backup**: Regular backups of models and data

### Version History:

- **v1.0.0**: Initial release with full system implementation
- Features: ETL pipeline, ML training, rule export, API, UI
- Technologies: Python, FastAPI, Streamlit, Next.js, XGBoost

---

Built with ❤️ using Python, FastAPI, Streamlit, Next.js, and Machine Learning
