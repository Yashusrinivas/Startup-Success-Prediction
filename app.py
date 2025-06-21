# Flask Application for Startup Prediction
# Following the project guide instructions

# Import necessary libraries (From your guide - Image 1)
from flask import Flask, render_template, request
import numpy as np

# Try to import required libraries and load model
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    print("✅ All required libraries imported successfully")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please install missing libraries using:")
    print("pip install scikit-learn joblib")
    exit(1)

# Load the saved machine learning model and preprocessing objects (From your guide - Image 2)
try:
    # Try to load the improved model first (better performance, less overfitting)
    model = joblib.load('improved_random_forest_model.pkl')
    print("✅ Improved model loaded successfully")
    model_type = "improved"
except FileNotFoundError:
    # Fallback to original model if improved model doesn't exist
    try:
        model = joblib.load('random_forest_model.pkl')
        print("✅ Original model loaded successfully")
        print("⚠️ Consider using the improved model for better performance")
        model_type = "original"
    except FileNotFoundError:
        print("❌ No model file found!")
        print("Available options:")
        print("  - improved_random_forest_model.pkl (recommended)")
        print("  - random_forest_model.pkl (original)")
        print("Please make sure at least one model file exists in the directory")
        exit(1)
except Exception as e:
    print(f"❌ Error loading improved model: {e}")
    exit(1)

# Try to load additional preprocessing objects if they exist
try:
    scaler = joblib.load('feature_scaler.pkl')
    print("✅ Feature scaler loaded successfully")
except:
    scaler = None
    print("⚠️ Feature scaler not found - using unscaled features")

try:
    feature_names = joblib.load('feature_names.pkl')
    print("✅ Feature names loaded successfully")
except:
    feature_names = None
    print("⚠️ Feature names not found - using default feature order")

# Try to load feature selector if using improved model with feature selection
try:
    feature_selector = joblib.load('feature_selector.pkl')
    print("✅ Feature selector loaded successfully")
    using_feature_selection = True
except:
    feature_selector = None
    using_feature_selection = False
    print("ℹ️ No feature selector found - using all features")

print(f"🤖 Model type: {model_type}")
print(f"⚖️ Using scaler: {'Yes' if scaler else 'No'}")
print(f"🎯 Using feature selection: {'Yes' if using_feature_selection else 'No'}")
print("=" * 50)

# Create Flask application instance (From your guide - Image 3)
app = Flask(__name__)


# Define route for homepage - Landing page (From your guide - Image 4)
@app.route('/')
def home():
    return render_template('home.html')


# Define route for prediction form
@app.route('/predict-form')
def predict_form():
    return render_template('index.html')


# Alternative route for prediction form (backward compatibility)
@app.route('/index')
def index():
    return render_template('index.html')


# Define route for prediction (From your guide - Image 5)
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form (From your guide - Image 5)
    age_first_funding_year = float(request.form['age_first_funding_year'])
    age_last_funding_year = float(request.form['age_last_funding_year'])
    age_first_milestone_year = float(request.form['age_first_milestone_year'])
    age_last_milestone_year = float(request.form['age_last_milestone_year'])
    relationships = float(request.form['relationships'])
    funding_rounds = float(request.form['funding_rounds'])
    funding_total_usd = float(request.form['funding_total_usd'])
    milestones = float(request.form['milestones'])
    avg_participants = float(request.form['avg_participants'])

    # Additional boolean/categorical features
    is_CA = int(request.form.get('is_CA', 0))
    is_NY = int(request.form.get('is_NY', 0))
    is_MA = int(request.form.get('is_MA', 0))
    is_TX = int(request.form.get('is_TX', 0))
    has_VC = int(request.form.get('has_VC', 0))
    has_angel = int(request.form.get('has_angel', 0))
    has_roundA = int(request.form.get('has_roundA', 0))
    has_roundB = int(request.form.get('has_roundB', 0))
    has_roundC = int(request.form.get('has_roundC', 0))
    has_roundD = int(request.form.get('has_roundD', 0))
    is_top500 = int(request.form.get('is_top500', 0))

    # Create feature array matching the exact model requirements
    if feature_names:
        # Create a full feature array with default values
        feature_dict = {}

        # Set default values for all expected features
        for feature in feature_names:
            if feature == 'latitude':
                # Default latitude based on state
                if is_CA:
                    feature_dict[feature] = 37.4419  # California
                elif is_NY:
                    feature_dict[feature] = 40.7128  # New York
                elif is_MA:
                    feature_dict[feature] = 42.3601  # Massachusetts
                elif is_TX:
                    feature_dict[feature] = 31.9686  # Texas
                else:
                    feature_dict[feature] = 39.8283  # US average
            elif feature == 'longitude':
                # Default longitude based on state
                if is_CA:
                    feature_dict[feature] = -122.1430  # California
                elif is_NY:
                    feature_dict[feature] = -74.0060  # New York
                elif is_MA:
                    feature_dict[feature] = -71.0589  # Massachusetts
                elif is_TX:
                    feature_dict[feature] = -99.9018  # Texas
                else:
                    feature_dict[feature] = -98.5795  # US average
            elif feature == 'age_first_funding_year':
                feature_dict[feature] = age_first_funding_year
            elif feature == 'age_last_funding_year':
                feature_dict[feature] = age_last_funding_year
            elif feature == 'age_first_milestone_year':
                feature_dict[feature] = age_first_milestone_year
            elif feature == 'age_last_milestone_year':
                feature_dict[feature] = age_last_milestone_year
            elif feature == 'relationships':
                feature_dict[feature] = relationships
            elif feature == 'funding_rounds':
                feature_dict[feature] = funding_rounds
            elif feature == 'funding_total_usd':
                feature_dict[feature] = funding_total_usd
            elif feature == 'milestones':
                feature_dict[feature] = milestones
            elif feature == 'avg_participants':
                feature_dict[feature] = avg_participants
            elif feature == 'is_CA':
                feature_dict[feature] = is_CA
            elif feature == 'is_NY':
                feature_dict[feature] = is_NY
            elif feature == 'is_MA':
                feature_dict[feature] = is_MA
            elif feature == 'is_TX':
                feature_dict[feature] = is_TX
            elif feature == 'is_otherstate':
                feature_dict[feature] = 1 if not any([is_CA, is_NY, is_MA, is_TX]) else 0
            elif feature == 'has_VC':
                feature_dict[feature] = has_VC
            elif feature == 'has_angel':
                feature_dict[feature] = has_angel
            elif feature == 'has_roundA':
                feature_dict[feature] = has_roundA
            elif feature == 'has_roundB':
                feature_dict[feature] = has_roundB
            elif feature == 'has_roundC':
                feature_dict[feature] = has_roundC
            elif feature == 'has_roundD':
                feature_dict[feature] = has_roundD
            elif feature == 'is_top500':
                feature_dict[feature] = is_top500
            elif feature == 'funding_per_round':
                feature_dict[feature] = funding_total_usd / (funding_rounds + 1)
            elif feature == 'founded_year':
                feature_dict[feature] = 2010  # Default founding year
            elif feature == 'company_age':
                feature_dict[feature] = 13  # Default company age (2023 - 2010)
            elif feature in ['state', 'category', 'state_category']:
                # Encoded categorical features
                if is_CA:
                    feature_dict[feature] = 0  # CA encoded as 0
                elif is_NY:
                    feature_dict[feature] = 1  # NY encoded as 1
                elif is_MA:
                    feature_dict[feature] = 2  # MA encoded as 2
                elif is_TX:
                    feature_dict[feature] = 3  # TX encoded as 3
                else:
                    feature_dict[feature] = 4  # Other encoded as 4
            else:
                # Default value for any other features
                feature_dict[feature] = 0

        # Create input array in the correct order
        input_array = np.array([feature_dict[feature] for feature in feature_names]).reshape(1, -1)
    else:
        # Fallback: create basic input array
        input_array = np.array([
            37.4419, -122.1430,  # Default lat/long (CA)
            age_first_funding_year, age_last_funding_year,
            age_first_milestone_year, age_last_milestone_year,
            relationships, funding_rounds, funding_total_usd, milestones,
            is_CA, is_NY, is_MA, is_TX, 1 if not any([is_CA, is_NY, is_MA, is_TX]) else 0,
            has_VC, has_angel, has_roundA, has_roundB, has_roundC, has_roundD,
            avg_participants, is_top500,
            2010, 0, 13,  # founded_year, state, company_age
            funding_total_usd / (funding_rounds + 1)  # funding_per_round
        ]).reshape(1, -1)

    # Apply scaling if scaler is available
    if scaler is not None:
        input_array = scaler.transform(input_array)

    # Make a prediction using the loaded model
    prediction = model.predict(input_array)[0]

    # Get prediction probability for confidence
    try:
        prediction_proba = model.predict_proba(input_array)[0]
        confidence = max(prediction_proba) * 100
    except:
        confidence = 85.0  # Default confidence if probability not available

    # Map the predicted label to a meaningful output
    if prediction == 1:
        result = 'Acquired'
        result_class = 'success'
        result_message = 'High probability of success! This startup shows strong indicators for acquisition.'
    else:
        result = 'Closed'
        result_class = 'danger'
        result_message = 'Higher risk of closure. Consider reviewing the business model and funding strategy.'

    # Create input summary for display
    input_summary = {
        'Age First Funding (Years)': age_first_funding_year,
        'Age Last Funding (Years)': age_last_funding_year,
        'Age First Milestone (Years)': age_first_milestone_year,
        'Age Last Milestone (Years)': age_last_milestone_year,
        'Relationships': relationships,
        'Funding Rounds': funding_rounds,
        'Total Funding (USD)': funding_total_usd,
        'Milestones': milestones,
        'Average Participants': avg_participants
    }

    # Render the prediction result
    return render_template('result.html',
                           result=result,
                           confidence=round(confidence, 1),
                           result_class=result_class,
                           result_message=result_message,
                           input_data=input_summary)


# Additional routes for other pages mentioned in guide
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


# Error handling
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


# Main function (From your guide)
if __name__ == '__main__':
    print("🚀 Starting Startup Prediction Flask Application...")
    print("📊 Model loaded successfully!")

    # Get port from environment variable (for Railway/Render deployment)
    import os

    port = int(os.environ.get('PORT', 5000))

    print(f"🌐 Access the application at port: {port}")
    print("💡 Use Ctrl+C to stop the server")

    # Run the Flask application (Railway compatible)
    app.run(debug=False, host='0.0.0.0', port=port)