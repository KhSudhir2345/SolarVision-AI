
# ☀️ Solar PV Loss Attribution System

## 📋 Project Overview

This project is a **comprehensive Solar PV Loss Attribution Analysis System** developed for the **Zelestra Hackathon**. It combines **physics-informed machine learning** with a modern **React dashboard** to analyze solar plant performance and categorize losses effectively.

---

## 🏗️ System Architecture

The system has **three core components**:

### 1. 🔧 Backend API Server (`app.py`)
- **Framework**: Flask
- **Purpose**: RESTful API server to handle file uploads, trigger ML analysis, and deliver results
- **Key Features**:
  - Secure session-based file management
  - Background ML processing with progress tracking
  - Result exporting (CSV, JSON, ZIP)
  - CORS support for frontend-backend communication

### 2. 🤖 ML Analysis Engine (`ml_utility.py`)
- **Frameworks**: Scikit-learn, XGBoost, Pandas, NumPy
- **Purpose**: Physics-informed ML for multi-level loss attribution
- **Key Features**:
  - Ensemble modeling with Random Forest and XGBoost
  - Physics-based feature engineering
  - Boolean flag detection at 15-minute intervals
  - 10+ types of solar loss classification

### 3. 🌐 Frontend Dashboard (`App.jsx`)
- **Framework**: React with TailwindCSS and Recharts
- **Purpose**: A modern dashboard for visualization, analysis control, and file handling
- **Key Features**:
  - Live progress tracking of ML analysis
  - Interactive data visualizations
  - AI-powered insights and charting
  - Direct file export support

---

## 🧪 Technical Stack

### 🔙 Backend
| Purpose             | Package                        |
|---------------------|--------------------------------|
| Core Framework      | Flask==2.3.3, flask-cors==4.0.0 |
| Data Processing     | pandas==2.1.1, numpy==1.24.3    |
| Machine Learning    | scikit-learn==1.3.0, xgboost==1.7.6, scipy==1.11.3 |
| Visualization       | matplotlib==3.7.2, seaborn==0.12.2, plotly==5.17.0 |
| Utilities           | joblib==1.3.2, werkzeug==2.3.7  |

### 🌐 Frontend (from package.json)
```
"dependencies": {
  "react": "^18.3.1",
  "react-dom": "^18.3.1",
  "recharts": "^2.8.0",
  "lucide-react": "^0.344.0",
  "axios": "^1.6.0"
},
"devDependencies": {
  "tailwindcss": "^3.4.1",
  "postcss": "^8.4.35",
  "autoprefixer": "^10.4.18",
  "vite": "^5.4.2",
  "@vitejs/plugin-react": "^4.3.1"
}
```

---

## 📁 File & Code Structure

### `app.py` – Backend API
**Endpoints**:
```
GET    /api/health               # Health check
POST   /api/upload               # Upload data
POST   /api/analyze/<id>         # Start analysis
GET    /api/status/<id>          # Get analysis progress
GET    /api/results/<id>         # Fetch results
GET    /api/download/<id>/<file> # Download specific result
GET    /api/download-all/<id>    # Download all results
GET    /api/sessions             # List all sessions
DELETE /api/session/<id>         # Delete session
GET    /api/config               # System config
```

### `ml_utility.py` – ML Engine
**Features**:
- Physics-based feature engineering (Solar Geometry, Irradiance, Temperature)
- Boolean flag classification of 10+ loss types
- Outputs: CSVs, JSON summaries, ZIP bundles

### `App.jsx` – React Frontend
**Features**:
- Upload, overview, loss analysis, performance, and AI insights tabs
- Charts: Area, Pie, Line, Radar, Bar (via Recharts)
- Export: CSV + ZIP

---

## 🚀 Live Demo

Try the deployed project here: (https://solar-vision-ai-git-main-khsudhir2345s-projects.vercel.app)


---

## ⚙️ For Developers

Follow these steps to run the project locally:

### 1. Clone the Repository
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

```
### 2. Set Up Frontend

```
cd frontend
npm install
npm run dev
```

### 3. Set Up Backend
```
cd backend
pip install -r requirements.txt
python app.py

```
> You can use a virtual environment if needed (`python -m venv venv`)

---

## 🔄 System Workflow

### 1. 📁 Data Upload
- User uploads CSV file through web interface.
- Backend validates file format and size.
- A session is created with a unique ID.
- File stored securely with timestamp.

### 2. ⚙️ Analysis Execution
- Background thread initiates ML analysis.
- Progress updates sent via polling.
- Physics-informed feature extraction.
- ML model training and prediction.
- Results compiled and exported.

### 3. 📊 Result Visualization
- Real-time dashboard updates.
- Interactive charts with actual and predicted data.
- AI-generated insights and recommendations.
- Export options for various formats.

### 4. 📤 Export & Download
- Multiple format options (CSV, JSON, ZIP).
- Zelestra-compliant deliverables.
- Session cleanup and management.

---

## 🎯 Contest Requirements Met

### ✅ Technical Requirements
- **Physics-Informed ML**: Feature engineering using solar physics principles.
- **Loss Attribution**: 10+ categorized loss mechanisms.
- **Boolean Flags**: Detection at 15-minute intervals.
- **Multi-Level Analysis**: Insights at plant, inverter, and string levels.
- **Visualization**: Real-time interactive dashboard.
- **Export Capability**: Support for multiple file formats.

### ✅ Performance Targets
- **Attribution Quality**: Less than 15% unattributed losses (Zelestra goal).
- **Model Accuracy**: R² > 90% for loss prediction.
- **Processing Efficiency**: Real-time progress tracking.
- **User Experience**: Modern, intuitive web UI.

---

## 📈 Future Enhancements

### 🔧 Potential Improvements
- Real-time data integration (e.g., SCADA).
- Weather API integration.
- Predictive maintenance models.
- Multi-plant comparative analysis.
- Native mobile app for field users.
- 3D visualization and heat maps.

### 🧩 Scalability Considerations
- Persistent database integration.
- Microservices architecture.
- Docker/Kubernetes containerization.
- Cloud deployment with auto-scaling.
- API rate limiting & authentication.

---

## 🔐 Environment Notes

- Currently, no API keys are required.
- Render/Vercel environment variables can be added for future enhancements.

---

## 📢 License

MIT License

---

> ✨ Built for the **Zelestra Hackathon**, combining deep technical insight with real-world solar applications.

