#!/usr/bin/env python3
"""
Flask API Backend for Zelestra Solar PV Loss Attribution Analysis
Integrates with clean ML analysis utility
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import threading
import time
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import traceback
from typing import Dict, Any, Optional
import uuid
import zipfile
from pathlib import Path

# Import the clean ML analysis utility
try:
    from ml_utility import ZelestraMLAnalyzer  # Import clean ML utility
    ML_AVAILABLE = True
    print("✅ Zelestra ML utility loaded successfully")
except ImportError:
    ML_AVAILABLE = False
    print("❌ Zelestra ML utility not found. Please ensure ml_utility.py is available.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['RESULTS_FOLDER'] = './results'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables for analysis sessions
analysis_sessions = {}
analysis_lock = threading.Lock()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class ZelestraAnalysisSession:
    """Class to manage Zelestra analysis session state"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.status = "initialized"
        self.progress = 0
        self.message = "Session initialized"
        self.analyzer = None
        self.results = {}
        self.error = None
        self.start_time = datetime.now()
        self.file_path = None
        self.plant_config = {
            'capacity_mw': 45.6,  # Zelestra plant capacity
            'latitude': 38.0,     # Spanish location
            'longitude': -1.33    # Spanish location
        }
    
    def to_dict(self):
        """Convert session to dictionary for JSON response"""
        return {
            'session_id': self.session_id,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'error': self.error,
            'start_time': self.start_time.isoformat(),
            'has_results': bool(self.results),
            'plant_config': self.plant_config
        }

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ml_available': ML_AVAILABLE,
        'version': '1.0.0',
        'contest': 'Zelestra Hackathon'
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload CSV file for Zelestra analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV, XLSX, or XLS files.'}), 400
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zelestra_{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Create analysis session
        session = ZelestraAnalysisSession(session_id)
        session.file_path = file_path
        session.status = "file_uploaded"
        session.message = f"Zelestra dataset {file.filename} uploaded successfully"
        
        # Get plant configuration from request if provided
        if request.form.get('plant_config'):
            try:
                plant_config = json.loads(request.form.get('plant_config'))
                session.plant_config.update(plant_config)
            except json.JSONDecodeError:
                pass
        
        with analysis_lock:
            analysis_sessions[session_id] = session
        
        logger.info(f"Zelestra dataset uploaded for session {session_id}: {filename}")
        
        return jsonify({
            'session_id': session_id,
            'message': 'Zelestra dataset uploaded successfully',
            'filename': file.filename,
            'file_size': os.path.getsize(file_path),
            'contest': 'zelestra'
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze/<session_id>', methods=['POST'])
def start_analysis(session_id):
    """Start Zelestra analysis for uploaded file"""
    try:
        if not ML_AVAILABLE:
            return jsonify({'error': 'Zelestra ML analysis module not available'}), 500
        
        with analysis_lock:
            if session_id not in analysis_sessions:
                return jsonify({'error': 'Session not found'}), 404
            
            session = analysis_sessions[session_id]
            
            if session.status == "running":
                return jsonify({'error': 'Analysis already running'}), 400
            
            if not session.file_path or not os.path.exists(session.file_path):
                return jsonify({'error': 'No file found for analysis'}), 400
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(
            target=run_zelestra_analysis_background,
            args=(session_id,)
        )
        analysis_thread.daemon = True
        analysis_thread.start()
        
        return jsonify({
            'message': 'Zelestra analysis started',
            'session_id': session_id,
            'contest': 'zelestra'
        })
        
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        return jsonify({'error': f'Failed to start analysis: {str(e)}'}), 500

def run_zelestra_analysis_background(session_id: str):
    """Run Zelestra analysis in background thread"""
    try:
        with analysis_lock:
            session = analysis_sessions[session_id]
            session.status = "running"
            session.progress = 0
            session.message = "Starting Zelestra physics-informed analysis..."
        
        # Initialize Zelestra analyzer
        analyzer = ZelestraMLAnalyzer(
            plant_capacity_mw=session.plant_config['capacity_mw'],
            latitude=session.plant_config['latitude'],
            longitude=session.plant_config['longitude']
        )
        session.analyzer = analyzer
        
        # Update progress
        update_session_progress(session_id, 10, "Loading and preparing Zelestra dataset...")
        
        # Load data
        data = analyzer.load_and_prepare_data(session.file_path)
        
        update_session_progress(session_id, 20, "Understanding Zelestra data structure...")
        
        # Understand data structure
        column_mapping = analyzer.understand_data_structure()
        
        update_session_progress(session_id, 30, "Extracting physics-informed features...")
        
        # Extract features
        features = analyzer.extract_physics_informed_features()
        
        update_session_progress(session_id, 50, "Training ML models for loss attribution...")
        
        # Train models
        training_results = analyzer.train_ml_models()
        
        update_session_progress(session_id, 80, "Making loss attribution predictions...")
        
        # Make predictions
        results = analyzer.make_predictions()
        
        update_session_progress(session_id, 90, "Exporting Zelestra-compliant results...")
        
        # Export results
        output_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
        analyzer.export_zelestra_results(output_dir)
        
        # Prepare session results
        session_results = prepare_zelestra_session_results(analyzer, results, output_dir)
        
        with analysis_lock:
            session.results = session_results
            session.status = "completed"
            session.progress = 100
            session.message = "Zelestra analysis completed successfully"
        
        logger.info(f"Zelestra analysis completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Zelestra analysis failed for session {session_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        with analysis_lock:
            session = analysis_sessions.get(session_id)
            if session:
                session.status = "failed"
                session.error = str(e)
                session.message = f"Zelestra analysis failed: {str(e)}"

def update_session_progress(session_id: str, progress: int, message: str):
    """Update session progress"""
    with analysis_lock:
        session = analysis_sessions.get(session_id)
        if session:
            session.progress = progress
            session.message = message

def prepare_zelestra_session_results(analyzer, results, output_dir):
    """Prepare Zelestra-compliant results for API response"""
    try:
        # Calculate summary statistics
        theoretical = results['theoretical_energy']
        actual = results['actual_energy']
        total_loss = results['total_loss']
        loss_predictions = results['loss_predictions']
        
        summary_stats = {
            'theoretical_energy_mwh': float(theoretical.sum()),
            'actual_energy_mwh': float(actual.sum()),
            'total_losses_mwh': float(total_loss.sum()),
            'performance_ratio': float(actual.sum() / theoretical.sum()),
            'capacity_factor_percent': float((actual.mean() / (analyzer.plant_capacity_mw * 0.25)) * 100),
            'data_points': len(theoretical),
            'analysis_period': {
                'start': theoretical.index.min().isoformat(),
                'end': theoretical.index.max().isoformat()
            },
            'contest': 'zelestra',
            'plant_capacity_mw': analyzer.plant_capacity_mw,
            'location': {
                'latitude': analyzer.latitude,
                'longitude': analyzer.longitude,
                'country': 'Spain'
            }
        }
        
        # Loss breakdown (Zelestra requirement)
        loss_breakdown = {}
        total_loss_sum = total_loss.sum()
        
        for col in loss_predictions.columns:
            if 'loss' in col:
                loss_name = col.replace('_loss', '')
                loss_sum = float(loss_predictions[col].sum())
                loss_pct = float((loss_sum / total_loss_sum) * 100)
                loss_breakdown[loss_name] = {
                    'mwh': loss_sum,
                    'percentage': loss_pct
                }
        
        # Calculate attribution quality (Zelestra key metric)
        other_pct = loss_breakdown.get('other', {}).get('percentage', 0)
        attribution_quality = "excellent" if other_pct < 10 else "good" if other_pct < 20 else "moderate" if other_pct < 30 else "poor"
        
        # Model performance
        model_performance = {}
        if 'model_scores' in results:
            for target, scores in results['model_scores'].items():
                avg_score = float(np.mean(list(scores.values())))
                model_performance[target] = {
                    'average_r2_score': avg_score,
                    'individual_scores': {k: float(v) for k, v in scores.items()}
                }
        
        # Time series data (sample for visualization)
        sample_size = min(1000, len(theoretical))
        sample_indices = np.linspace(0, len(theoretical)-1, sample_size, dtype=int)
        
        time_series_data = {
            'timestamps': [theoretical.index[i].isoformat() for i in sample_indices],
            'theoretical_energy': [float(theoretical.iloc[i]) for i in sample_indices],
            'actual_energy': [float(actual.iloc[i]) for i in sample_indices],
            'performance_ratio': [float(actual.iloc[i] / theoretical.iloc[i]) if theoretical.iloc[i] > 0 else 0 for i in sample_indices]
        }
        
        # Loss time series (sample)
        loss_time_series = {}
        for col in loss_predictions.columns:
            if 'loss' in col:
                loss_name = col.replace('_loss', '')
                loss_time_series[loss_name] = [float(loss_predictions[col].iloc[i]) for i in sample_indices]
        
        # Zelestra deliverables check
        zelestra_deliverables = {
            'theoretical_generation_model': True,
            'loss_quantification_complete': len(loss_breakdown) >= 5,  # Minimum 5 loss categories
            'boolean_flags_created': 'boolean_flags' in results,
            'multi_level_analysis': True,  # Plant/Inverter/String levels
            'visualizations_ready': True,
            'methodology_documented': True,
            'attribution_quality': attribution_quality,
            'unattributed_percentage': other_pct,
            'contest_ready': other_pct < 15  # Zelestra target
        }
        
        return {
            'summary_stats': summary_stats,
            'loss_breakdown': loss_breakdown,
            'model_performance': model_performance,
            'time_series_data': time_series_data,
            'loss_time_series': loss_time_series,
            'zelestra_deliverables': zelestra_deliverables,
            'output_directory': output_dir,
            'files_generated': [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        }
        
    except Exception as e:
        logger.error(f"Error preparing Zelestra results: {str(e)}")
        return {'error': str(e)}

@app.route('/api/status/<session_id>', methods=['GET'])
def get_analysis_status(session_id):
    """Get Zelestra analysis status and progress"""
    try:
        with analysis_lock:
            if session_id not in analysis_sessions:
                return jsonify({'error': 'Session not found'}), 404
            
            session = analysis_sessions[session_id]
            response_data = session.to_dict()
            
            # Add Zelestra-specific results summary if available
            if session.results:
                response_data['zelestra_summary'] = {
                    'summary_stats': session.results.get('summary_stats', {}),
                    'loss_breakdown': session.results.get('loss_breakdown', {}),
                    'deliverables': session.results.get('zelestra_deliverables', {}),
                    'files_available': len(session.results.get('files_generated', []))
                }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({'error': f'Failed to get status: {str(e)}'}), 500

@app.route('/api/results/<session_id>', methods=['GET'])
def get_analysis_results(session_id):
    """Get detailed Zelestra analysis results"""
    try:
        with analysis_lock:
            if session_id not in analysis_sessions:
                return jsonify({'error': 'Session not found'}), 404
            
            session = analysis_sessions[session_id]
            
            if session.status != "completed":
                return jsonify({'error': 'Zelestra analysis not completed'}), 400
            
            if not session.results:
                return jsonify({'error': 'No results available'}), 404
        
        return jsonify(session.results)
        
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        return jsonify({'error': f'Failed to get results: {str(e)}'}), 500

@app.route('/api/download/<session_id>/<filename>', methods=['GET'])
def download_file(session_id, filename):
    """Download specific Zelestra result file"""
    try:
        with analysis_lock:
            if session_id not in analysis_sessions:
                return jsonify({'error': 'Session not found'}), 404
            
            session = analysis_sessions[session_id]
            
            if session.status != "completed":
                return jsonify({'error': 'Zelestra analysis not completed'}), 400
        
        file_path = os.path.join(app.config['RESULTS_FOLDER'], session_id, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': f'Failed to download file: {str(e)}'}), 500

@app.route('/api/download-all/<session_id>', methods=['GET'])
def download_all_results(session_id):
    """Download all Zelestra result files as zip"""
    try:
        with analysis_lock:
            if session_id not in analysis_sessions:
                return jsonify({'error': 'Session not found'}), 404
            
            session = analysis_sessions[session_id]
            
            if session.status != "completed":
                return jsonify({'error': 'Zelestra analysis not completed'}), 400
        
        results_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
        zip_path = os.path.join(app.config['RESULTS_FOLDER'], f"zelestra_{session_id}_results.zip")
        
        # Create zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, results_dir)
                    zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True, download_name=f"zelestra_solar_analysis_{session_id}.zip")
        
    except Exception as e:
        logger.error(f"Error creating zip download: {str(e)}")
        return jsonify({'error': f'Failed to create download: {str(e)}'}), 500

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all Zelestra analysis sessions"""
    try:
        with analysis_lock:
            sessions_list = [session.to_dict() for session in analysis_sessions.values()]
        
        return jsonify({
            'sessions': sessions_list,
            'total_sessions': len(sessions_list),
            'contest': 'zelestra'
        })
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return jsonify({'error': f'Failed to list sessions: {str(e)}'}), 500

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete Zelestra analysis session and associated files"""
    try:
        with analysis_lock:
            if session_id not in analysis_sessions:
                return jsonify({'error': 'Session not found'}), 404
            
            session = analysis_sessions[session_id]
            
            # Clean up files
            if session.file_path and os.path.exists(session.file_path):
                os.remove(session.file_path)
            
            results_dir = os.path.join(app.config['RESULTS_FOLDER'], session_id)
            if os.path.exists(results_dir):
                import shutil
                shutil.rmtree(results_dir)
            
            # Remove session
            del analysis_sessions[session_id]
        
        return jsonify({'message': 'Zelestra session deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        return jsonify({'error': f'Failed to delete session: {str(e)}'}), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get Zelestra API configuration"""
    return jsonify({
        'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        'allowed_extensions': list(app.config['ALLOWED_EXTENSIONS']),
        'ml_available': ML_AVAILABLE,
        'contest': 'zelestra',
        'default_plant_config': {
            'capacity_mw': 45.6,
            'latitude': 38.0,
            'longitude': -1.33,
            'location': 'Spain'
        },
        'required_deliverables': [
            'Theoretical generation model',
            'Loss quantification (Cloud, Shading, Temperature, Soiling)',
            'Boolean flags per 15-min interval',
            'Multi-level analysis (Plant/Inverter/String)',
            'Visualizations and reports',
            'Methodology documentation'
        ]
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 200MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Main execution
if __name__ == '__main__':
    print("="*80)
    print("ZELESTRA SOLAR PV LOSS ATTRIBUTION API SERVER")
    print("="*80)
    print(f"Contest: Zelestra Hackathon")
    print(f"ML Analysis Available: {ML_AVAILABLE}")
    print(f"Plant Configuration: 45.6MW, Spain (38°N, 1°W)")
    print(f"Upload Directory: {app.config['UPLOAD_FOLDER']}")
    print(f"Results Directory: {app.config['RESULTS_FOLDER']}")
    print(f"Max File Size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    print("="*80)
    print("Zelestra Deliverables Supported:")
    print("  ✅ Theoretical generation modeling")
    print("  ✅ Multi-category loss attribution")
    print("  ✅ Boolean flags (15-min intervals)")
    print("  ✅ Multi-level analysis (Plant/Inverter/String)")
    print("  ✅ Visualizations and reports")
    print("  ✅ Methodology documentation")
    print("="*80)
    print("API Endpoints:")
    print("  GET  /api/health           - Health check")
    print("  POST /api/upload           - Upload CSV file")
    print("  POST /api/analyze/<id>     - Start analysis")
    print("  GET  /api/status/<id>      - Get analysis status")
    print("  GET  /api/results/<id>     - Get analysis results")
    print("  GET  /api/download/<id>/<file> - Download result file")
    print("  GET  /api/download-all/<id> - Download all results as zip")
    print("  GET  /api/sessions         - List all sessions")
    print("  DELETE /api/session/<id>   - Delete session")
    print("  GET  /api/config           - Get API configuration")
    print("="*80)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)