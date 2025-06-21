# 🚀 Startup Success Prediction

A machine learning-powered web application that predicts whether a startup will be **acquired** or **closed** based on historical data and key business metrics.

## 📊 Project Overview

This project leverages a Random Forest Classifier to analyze startup success patterns using funding data, geographic factors, and business metrics. The model is integrated into a user-friendly Flask web application, enabling entrepreneurs and investors to make data-driven decisions.

### 🎯 Key Features

- **78% Prediction Accuracy** with low overfitting
- **Interactive Web Interface** for real-time predictions
- **Actionable Recommendations** based on prediction outcomes
- **Comprehensive Analysis** of 923 startup records with 50+ features

## 🏆 Project Objectives

- **Primary Goal**: Develop a high-accuracy ML model for startup success prediction
- **Secondary Goals**:
  - Perform exploratory data analysis to uncover success factors
  - Build an interactive Flask web application
  - Provide actionable recommendations for startup strategies

## 📈 Dataset Description

The dataset contains **923 startup records** with **50 features** across multiple categories:

| Category | Features | Description |
|----------|----------|-------------|
| **Geographic** | `state_code`, `latitude`, `longitude`, `is_CA`, `is_NY` | Location-based indicators with binary flags |
| **Funding** | `funding_rounds`, `funding_total_usd`, `has_roundA`, `has_roundB` | Funding stages and capital raised |
| **Timeline** | `age_first_funding_year`, `age_last_funding_year`, `founded_at` | Temporal metrics and milestones |
| **Business Metrics** | `relationships`, `milestones`, `avg_participants` | Team connections and achievements |
| **Categorical** | `category_code`, `is_software`, `is_web` | Industry classifications |
| **Target** | `status` (acquired/closed) | Binary outcome (1=acquired, 0=closed) |

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- **Tools**: Pandas, NumPy, Seaborn, Matplotlib
- **Key Insights**:
  - California-based startups dominate the dataset
  - Early funding and strong relationships correlate with acquisition
  - Higher funding rounds increase success probability

### 2. Model Development
- **Algorithm**: Random Forest Classifier
- **Preprocessing**: One-hot encoding, feature scaling, 80/20 train-test split
- **Hyperparameter Tuning**: GridSearchCV optimization

### 3. Overfitting Mitigation
Applied 5 strategies to achieve optimal performance:

| Strategy | Training Acc | Testing Acc | Overfitting |
|----------|--------------|-------------|-------------|
| Conservative Hyperparameters | 85.14% | 77.62% | **7.52%** ✅ |
| Feature Selection | 87.31% | 76.53% | 10.77% |
| Manual Regularization | 85.14% | 76.90% | 8.24% |

## 📊 Model Performance

### Final Model Metrics
- **Training Accuracy**: 85.14%
- **Testing Accuracy**: 77.62%
- **Cross-Validation Score**: 80.19%

### Classification Report
```
              precision    recall  f1-score   support
0 (Closed)        0.85      0.45      0.59        98
1 (Acquired)      0.76      0.96      0.85       179
accuracy                            0.78       277
macro avg         0.80      0.70      0.72       277
weighted avg      0.79      0.78      0.75       277
```

## 🌐 Flask Web Application

### Application Structure
- **Backend** (`app.py`): Model loading, data preprocessing, prediction logic
- **Frontend**: 
  - `home.html`: Landing page with platform overview
  - `index.html`: Interactive prediction form
  - `result.html`: Results display with recommendations

### Key Features
- **User Input**: 20+ features including funding, milestones, location
- **Smart Predictions**: Confidence scores and detailed analysis
- **Actionable Insights**: Tailored recommendations for success/improvement
- **Responsive Design**: Mobile-friendly interface

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git
- Web browser

### Installation

1. **Clone the repository**
```bash
git clone <repository_url>
cd startup-success-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure model files are present**
- `improved_random_forest_model.pkl`
- `feature_scaler.pkl`
- `feature_names.pkl`

4. **Run the application**
```bash
python app.py
```

5. **Access the app**
Navigate to `http://localhost:5000`

### Dependencies
```txt
flask==2.0.1
numpy==1.26.4
scikit-learn==1.2.2
joblib==1.2.0
```

## 🔍 Usage

1. **Homepage**: Explore platform features and model statistics
2. **Prediction Form**: Input startup details via `/predict-form`
3. **Results**: View predictions, confidence scores, and recommendations
4. **Iterate**: Test different scenarios with the "Make Another Prediction" feature

## 💡 Key Insights

### Success Factors
- **Top Predictors**: `age_first_funding_year`, `relationships`, `funding_total_usd`, `milestones`
- **Geographic Advantage**: California-based startups show higher success rates
- **Funding Impact**: Early funding and multiple rounds increase acquisition probability

### Practical Applications
- **For Entrepreneurs**: Identify areas for improvement and strategic focus
- **For Investors**: Data-driven investment decision support
- **For Researchers**: Baseline model for startup success analysis

## ⚠️ Limitations

- **Data Scope**: Limited to 923 U.S.-based startups
- **Static Features**: Lacks real-time market data
- **Qualitative Factors**: Missing team quality and market fit metrics

## 🔮 Future Improvements

- [ ] Integrate real-time market data via APIs
- [ ] Expand dataset to include global startups
- [ ] Add feature importance visualizations
- [ ] Develop mobile app version
- [ ] Include qualitative assessment metrics

## 🛠️ Technical Skills Demonstrated

**Machine Learning**: Random Forest, Hyperparameter Tuning, Cross-Validation, Feature Engineering
**Web Development**: Flask, HTML, Responsive Design
**Data Science**: Pandas, NumPy, Scikit-learn, EDA, Data Visualization
**Soft Skills**: Problem-Solving, Project Management, Technical Communication

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

**[Yasaswini Locharla]** - Lead Developer and Data Scientist
- Email: yashusrinivas190@gmail.com

---

⭐ **Star this repository if you find it helpful!**

*Made with ❤️ for the startup community*
