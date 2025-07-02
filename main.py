import streamlit as st
import PyPDF2
import pdf2image
from PIL import Image
import pytesseract
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
import platform
from dotenv import load_dotenv
import re
from typing import Dict, List, Any
import requests
from datetime import datetime

# Load environment variables
load_dotenv()

# Windows Configuration
def setup_windows_paths():
    """Configure paths for Windows"""
    if platform.system() == "Windows":
        # Tesseract paths
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
            r'C:\tesseract\tesseract.exe'
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"Found Tesseract at: {path}")
                break
        else:
            st.error("Tesseract not found. Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        
        # Poppler paths
        poppler_paths = [
            r'C:\poppler\Library\bin',
            r'C:\Program Files\poppler\Library\bin',
            r'C:\Program Files (x86)\poppler\Library\bin'
        ]
        
        current_path = os.environ.get('PATH', '')
        for path in poppler_paths:
            if os.path.exists(path) and path not in current_path:
                os.environ['PATH'] = path + os.pathsep + current_path
                print(f"Added Poppler to PATH: {path}")
                break

# Initialize paths
setup_windows_paths()

class MedicalPDFAnalyzer:
    def __init__(self):
        # Load OpenRouter API key from environment
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            st.error("OpenRouter API key not found. Please set OPENROUTER_API_KEY in your .env file")
            st.info("Create a .env file with: OPENROUTER_API_KEY=your_openrouter_api_key_here")
            st.stop()
        
        self.model_loaded = True  # No local model loading required
        st.success("OpenRouter API configured successfully!")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF using PyPDF2 first, then OCR for images"""
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text + "\n"
            
            if len(text.strip()) < 100:
                st.info("Little text found in PDF, using OCR for scanned images...")
                return self.extract_text_with_ocr(pdf_file)
            
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_text_with_ocr(self, pdf_file) -> str:
        """Extract text from scanned PDF using OCR"""
        try:
            pdf_file.seek(0)
            with st.spinner("Converting PDF to images for OCR..."):
                images = pdf2image.convert_from_bytes(pdf_file.read(), dpi=300)
            
            full_text = ""
            progress_bar = st.progress(0)
            
            for i, image in enumerate(images):
                progress_bar.progress((i + 1) / len(images))
                text = pytesseract.image_to_string(image, config='--psm 6')
                full_text += f"Page {i+1}:\n{text}\n\n"
            
            progress_bar.empty()
            return full_text
        except Exception as e:
            st.error(f"Error with OCR extraction: {str(e)}")
            return ""
    
    def analyze_medical_data(self, text: str) -> Dict[str, Any]:
        """Analyze medical data using Llama 3 via OpenRouter with comprehensive extraction"""
        if not self.model_loaded:
            st.error("OpenRouter API not configured")
            return {}
        
        try:
            # Truncate text if too long for model context
            max_length = 3000  # Adjust based on OpenRouter's limits
            if len(text) > max_length:
                text = text[:max_length]
                st.warning(f"Text truncated to {max_length} characters for analysis")
            
            prompt = f"""
            Analyze the following medical document and extract ALL available information in JSON format. 
            If any information is not found in the document, mark it as "N/A".
            
            Medical Document Text:
            {text}
            
            Provide a comprehensive analysis in this JSON structure:
            {{
                "document_metadata": {{
                    "extraction_date": "{datetime.now().isoformat()}",
                    "document_type": "medical_report",
                    "file_source": "uploaded_pdf",
                    "analysis_confidence": "high/medium/low"
                }},
                "administrative_info": {{
                    "bill_number": "extracted or N/A",
                    "mr_number": "medical record number or N/A",
                    "room_ward_number": "room/ward number or N/A",
                    "hospital_name": "extracted or N/A",
                    "hospital_address": "extracted or N/A",
                    "hospital_phone": "extracted or N/A",
                    "department": "extracted or N/A",
                    "admission_number": "extracted or N/A"
                }},
                "patient_info": {{
                    "name": "extracted or N/A",
                    "age": "extracted or N/A",
                    "gender": "extracted or N/A",
                    "date_of_birth": "extracted or N/A",
                    "address": "extracted or N/A",
                    "phone_number": "extracted or N/A",
                    "emergency_contact": "extracted or N/A",
                    "insurance_info": "extracted or N/A",
                    "patient_id": "extracted or N/A"
                }},
                "visit_details": {{
                    "date_of_visit": "extracted or N/A",
                    "admission_date": "extracted or N/A",
                    "discharge_date": "extracted or N/A",
                    "visit_type": "outpatient/inpatient/emergency or N/A",
                    "chief_complaint": "extracted or N/A",
                    "referring_physician": "extracted or N/A"
                }},
                "medical_staff": {{
                    "attending_physician": "extracted or N/A",
                    "consultant_name": "extracted or N/A",
                    "resident_doctor": "extracted or N/A",
                    "nurse_in_charge": "extracted or N/A",
                    "other_staff": []
                }},
                "vital_signs": {{
                    "blood_pressure_systolic": "number or N/A",
                    "blood_pressure_diastolic": "number or N/A",
                    "heart_rate": "number or N/A",
                    "temperature": "number or N/A",
                    "respiratory_rate": "number or N/A",
                    "oxygen_saturation": "number or N/A",
                    "weight": "number or N/A",
                    "height": "number or N/A",
                    "bmi": "calculated or N/A",
                    "pain_scale": "1-10 or N/A"
                }},
                "lab_results": [
                    {{
                        "test_name": "name",
                        "value": "numeric value with unit",
                        "normal_range": "range",
                        "status": "normal/abnormal/critical",
                        "date": "test date or N/A"
                    }}
                ],
                "medications": [
                    {{
                        "name": "medication name",
                        "dosage": "strength",
                        "frequency": "how often",
                        "route": "oral/IV/etc",
                        "start_date": "date or N/A",
                        "duration": "how long or N/A",
                        "prescribing_doctor": "doctor name or N/A"
                    }}
                ],
                "procedures": [
                    {{
                        "name": "procedure name",
                        "date": "when performed",
                        "doctor": "who performed",
                        "outcome": "result",
                        "complications": "any issues or none"
                    }}
                ],
                "diagnoses": [
                    {{
                        "primary_diagnosis": "main diagnosis",
                        "secondary_diagnoses": [],
                        "icd_code": "code if available or N/A",
                        "diagnosis_date": "date or N/A",
                        "severity": "mild/moderate/severe or N/A"
                    }}
                ],
                "imaging_studies": [
                    {{
                        "type": "X-ray/CT/MRI/etc",
                        "body_part": "area examined",
                        "date": "when done",
                        "findings": "what was found",
                        "radiologist": "who read it or N/A"
                    }}
                ],
                "appointments_schedule": [
                    {{
                        "type": "follow-up/procedure/consultation",
                        "date": "scheduled date",
                        "time": "scheduled time",
                        "doctor": "with whom",
                        "purpose": "reason for appointment",
                        "location": "where"
                    }}
                ],
                "doctor_recommendations": [],
                "discharge_instructions": [],
                "key_findings": [],
                "risk_factors": [],
                "allergies": [],
                "medical_history": [],
                "family_history": [],
                "social_history": {{
                    "smoking": "yes/no/former or N/A",
                    "alcohol": "frequency or N/A",
                    "occupation": "job or N/A",
                    "exercise": "frequency or N/A"
                }},
                "follow_up_required": "details or N/A",
                "billing_info": {{
                    "total_charges": "amount or N/A",
                    "insurance_coverage": "amount or N/A",
                    "patient_responsibility": "amount or N/A",
                    "payment_status": "paid/pending/partial or N/A"
                }},
                "chart_data": {{
                    "trend_analysis": [],
                    "comparison_data": [],
                    "time_series": []
                }}
            }}
            
            Extract ALL numerical values that could be used for charts and graphs.
            """
            
            with st.spinner("Analyzing with Llama 3 via OpenRouter (this may take a moment)..."):
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "meta-llama/llama-3.1-8b-instruct",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 3000,
                    "temperature": 0.1
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    st.error(f"OpenRouter API error: {response.text}")
                    return {}
                
                analysis_text = response.json()["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
                
        except Exception as e:
            st.error(f"Error analyzing medical data: {str(e)}")
            return {}
    
    def create_comprehensive_charts(self, analysis_data: Dict[str, Any]) -> List[go.Figure]:
        """Create comprehensive visualizations based on the analysis"""
        charts = []
        
        try:
            # 1. Vital Signs Dashboard
            if analysis_data.get("vital_signs"):
                vital_signs = analysis_data["vital_signs"]
                vital_data = []
                
                for key, value in vital_signs.items():
                    if value and value != "N/A" and str(value).replace('.', '').isdigit():
                        vital_data.append({
                            "Parameter": key.replace("_", " ").title(), 
                            "Value": float(value),
                            "Unit": self.get_vital_sign_unit(key)
                        })
                
                if vital_data:
                    df_vitals = pd.DataFrame(vital_data)
                    fig_vitals = px.bar(df_vitals, x="Parameter", y="Value", 
                                      title="📊 Vital Signs Overview",
                                      color="Parameter",
                                      text="Value")
                    fig_vitals.update_traces(texttemplate='%{text}', textposition='outside')
                    fig_vitals.update_layout(showlegend=False, height=400)
                    charts.append(fig_vitals)
            
            # 2. Lab Results Visualization
            if analysis_data.get("lab_results"):
                lab_results = analysis_data["lab_results"]
                lab_data = []
                
                for result in lab_results:
                    if isinstance(result, dict) and result.get("value"):
                        numeric_value = self.extract_numeric_value(str(result["value"]))
                        if numeric_value:
                            lab_data.append({
                                "Test": result.get("test_name", "Unknown"),
                                "Value": numeric_value,
                                "Status": result.get("status", "Unknown"),
                                "Normal Range": result.get("normal_range", "N/A")
                            })
                
                if lab_data:
                    df_labs = pd.DataFrame(lab_data)
                    fig_labs = px.bar(df_labs, x="Test", y="Value", color="Status",
                                    title="🔬 Laboratory Results",
                                    color_discrete_map={
                                        "normal": "#28a745",
                                        "abnormal": "#ffc107", 
                                        "critical": "#dc3545"
                                    })
                    fig_labs.update_layout(height=400)
                    charts.append(fig_labs)
            
            # 3. Medications Pie Chart
            if analysis_data.get("medications"):
                medications = analysis_data["medications"]
                if medications:
                    med_data = []
                    for med in medications[:10]:  # Limit to top 10
                        if isinstance(med, dict):
                            name = med.get("name", "Unknown")
                            dosage = med.get("dosage", "")
                            med_data.append(f"{name} ({dosage})")
                        else:
                            med_data.append(str(med))
                    
                    if med_data:
                        fig_meds = go.Figure(data=[go.Pie(
                            labels=med_data,
                            values=[1] * len(med_data),
                            title="💊 Current Medications Distribution"
                        )])
                        fig_meds.update_layout(height=400)
                        charts.append(fig_meds)
            
            # 4. Blood Pressure Chart (if both systolic and diastolic available)
            vital_signs = analysis_data.get("vital_signs", {})
            if (vital_signs.get("blood_pressure_systolic", "N/A") != "N/A" and 
                vital_signs.get("blood_pressure_diastolic", "N/A") != "N/A"):
                
                systolic = float(vital_signs["blood_pressure_systolic"])
                diastolic = float(vital_signs["blood_pressure_diastolic"])
                
                fig_bp = go.Figure()
                fig_bp.add_trace(go.Bar(
                    x=['Systolic', 'Diastolic'],
                    y=[systolic, diastolic],
                    marker_color=['#ff6b6b', '#4ecdc4'],
                    text=[f'{systolic} mmHg', f'{diastolic} mmHg'],
                    textposition='auto'
                ))
                fig_bp.update_layout(
                    title="🫀 Blood Pressure Reading",
                    yaxis_title="mmHg",
                    height=400
                )
                charts.append(fig_bp)
            
            # 5. Procedures Timeline (if dates available)
            if analysis_data.get("procedures"):
                procedures = analysis_data["procedures"]
                timeline_data = []
                
                for proc in procedures:
                    if isinstance(proc, dict) and proc.get("date", "N/A") != "N/A":
                        timeline_data.append({
                            "Procedure": proc.get("name", "Unknown"),
                            "Date": proc.get("date"),
                            "Doctor": proc.get("doctor", "N/A"),
                            "Outcome": proc.get("outcome", "N/A")
                        })
                
                if timeline_data:
                    df_timeline = pd.DataFrame(timeline_data)
                    fig_timeline = px.timeline(
                        df_timeline, 
                        x_start="Date", 
                        x_end="Date",
                        y="Procedure",
                        title="📅 Procedures Timeline",
                        color="Outcome"
                    )
                    fig_timeline.update_layout(height=400)
                    charts.append(fig_timeline)
            
            # 6. Risk Factors Analysis
            if analysis_data.get("risk_factors"):
                risk_factors = analysis_data["risk_factors"]
                if risk_factors:
                    risk_counts = {}
                    for risk in risk_factors:
                        risk_type = str(risk).split()[0] if risk else "Other"
                        risk_counts[risk_type] = risk_counts.get(risk_type, 0) + 1
                    
                    if risk_counts:
                        fig_risks = go.Figure(data=[go.Pie(
                            labels=list(risk_counts.keys()),
                            values=list(risk_counts.values()),
                            title="⚠️ Risk Factors Distribution"
                        )])
                        fig_risks.update_traces(marker_colors=['#ff9999', '#ffcc99', '#99ccff', '#cc99ff'])
                        fig_risks.update_layout(height=400)
                        charts.append(fig_risks)
            
        except Exception as e:
            st.error(f"Error creating charts: {str(e)}")
        
        return charts
    
    def get_vital_sign_unit(self, parameter: str) -> str:
        """Get appropriate unit for vital sign parameter"""
        units = {
            "blood_pressure_systolic": "mmHg",
            "blood_pressure_diastolic": "mmHg", 
            "heart_rate": "bpm",
            "temperature": "°F",
            "respiratory_rate": "/min",
            "oxygen_saturation": "%",
            "weight": "lbs",
            "height": "in",
            "bmi": "kg/m²",
            "pain_scale": "/10"
        }
        return units.get(parameter, "")
    
    def extract_numeric_value(self, text: str) -> float:
        """Extract numeric value from text"""
        try:
            match = re.search(r'(\d+\.?\d*)', text)
            if match:
                return float(match.group(1))
        except:
            pass
        return None
    
    def create_summary_statistics(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for the medical data"""
        stats = {
            "total_medications": len(analysis_data.get("medications", [])),
            "total_procedures": len(analysis_data.get("procedures", [])),
            "total_lab_tests": len(analysis_data.get("lab_results", [])),
            "total_diagnoses": len(analysis_data.get("diagnoses", [])),
            "total_appointments": len(analysis_data.get("appointments_schedule", [])),
            "risk_factor_count": len(analysis_data.get("risk_factors", [])),
        }
        return stats

def main():
    st.set_page_config(
        page_title="Enhanced Medical PDF Analyzer",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Enhanced Medical PDF Analysis with Comprehensive Data Extraction")
    st.markdown("Upload medical PDFs to extract comprehensive data and generate AI-powered insights with detailed visualizations")
    
    # Initialize analyzer
    analyzer = MedicalPDFAnalyzer()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check Tesseract
        try:
            version = pytesseract.get_tesseract_version()
            st.success(f"✅ Tesseract: {version}")
        except:
            st.error("❌ Tesseract not found")
        
        # Check API
        if analyzer.model_loaded:
            st.success("✅ OpenRouter API configured")
        else:
            st.error("❌ API not configured")
        
        st.header("📊 Features")
        st.info("""
        **Enhanced Extraction:**
        - Bill #, MR #, Room/Ward #
        - Patient & Hospital Details
        - Medical Staff Information
        - Comprehensive Vital Signs
        - Lab Results & Medications
        - Procedures & Appointments
        - Risk Factors & Recommendations
        
        **Advanced Visualizations:**
        - Interactive Charts & Graphs
        - Trend Analysis
        - Risk Assessment
        - Timeline Views
        """)
        
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Medical PDF", 
        type=['pdf'],
        help="Upload scanned or text-based medical PDFs for comprehensive analysis"
    )
    
    if uploaded_file is not None:
        # Show file details
        file_details = {
            "📄 Filename": uploaded_file.name,
            "📏 File size": f"{uploaded_file.size / 1024:.2f} KB",
            "⏰ Upload time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.write("**File Details:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")
        with col3:
            st.metric("Status", "✅ Ready")
        
        with st.spinner("🔄 Processing PDF..."):
            # Extract text
            text = analyzer.extract_text_from_pdf(uploaded_file)
            
            if text:
                st.success("✅ Text extracted successfully!")
                
                with st.expander("👀 View Extracted Text"):
                    st.text_area("Raw Extracted Text", text, height=200)
                
                with st.spinner("🤖 Analyzing medical data with AI..."):
                    # Analyze with comprehensive extraction
                    analysis = analyzer.analyze_medical_data(text)
                    
                    if analysis:
                        st.success("✅ Comprehensive analysis completed!")
                        
                        # Create summary statistics
                        stats = analyzer.create_summary_statistics(analysis)
                        
                        # Display summary metrics
                        st.header("📈 Summary Statistics")
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        
                        with col1:
                            st.metric("💊 Medications", stats["total_medications"])
                        with col2:
                            st.metric("🔬 Lab Tests", stats["total_lab_tests"])
                        with col3:
                            st.metric("⚕️ Procedures", stats["total_procedures"])
                        with col4:
                            st.metric("🩺 Diagnoses", stats["total_diagnoses"])
                        with col5:
                            st.metric("📅 Appointments", stats["total_appointments"])
                        with col6:
                            st.metric("⚠️ Risk Factors", stats["risk_factor_count"])
                        
                        # Display results in tabs
                        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                            "📊 Charts & Analytics", 
                            "🏥 Administrative Info",
                            "👤 Patient Details", 
                            "💊 Medical Data", 
                            "📋 Recommendations",
                            "📥 Download Data"
                        ])
                        
                        with tab1:
                            st.header("📊 Comprehensive Medical Data Visualizations")
                            charts = analyzer.create_comprehensive_charts(analysis)
                            
                            if charts:
                                # Display charts in a grid
                                for i, chart in enumerate(charts):
                                    st.plotly_chart(chart, use_container_width=True)
                                    if i < len(charts) - 1:
                                        st.divider()
                            else:
                                st.info("📈 No chartable data found in the document")
                        
                        with tab2:
                            st.header("🏥 Administrative Information")
                            
                            if analysis.get("administrative_info"):
                                admin_info = analysis["administrative_info"]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("🏥 Hospital Details")
                                    st.write(f"**Hospital:** {admin_info.get('hospital_name', 'N/A')}")
                                    st.write(f"**Address:** {admin_info.get('hospital_address', 'N/A')}")
                                    st.write(f"**Phone:** {admin_info.get('hospital_phone', 'N/A')}")
                                    st.write(f"**Department:** {admin_info.get('department', 'N/A')}")
                                
                                with col2:
                                    st.subheader("📄 Document Details")
                                    st.write(f"**Bill #:** {admin_info.get('bill_number', 'N/A')}")
                                    st.write(f"**MR #:** {admin_info.get('mr_number', 'N/A')}")
                                    st.write(f"**Room/Ward #:** {admin_info.get('room_ward_number', 'N/A')}")
                                    st.write(f"**Admission #:** {admin_info.get('admission_number', 'N/A')}")
                            
                            if analysis.get("medical_staff"):
                                st.subheader("👨‍⚕️ Medical Staff")
                                staff = analysis["medical_staff"]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Attending Physician:** {staff.get('attending_physician', 'N/A')}")
                                    st.write(f"**Consultant:** {staff.get('consultant_name', 'N/A')}")
                                with col2:
                                    st.write(f"**Resident Doctor:** {staff.get('resident_doctor', 'N/A')}")
                                    st.write(f"**Nurse in Charge:** {staff.get('nurse_in_charge', 'N/A')}")
                        
                        with tab3:
                            st.header("👤 Patient Information")
                            
                            if analysis.get("patient_info"):
                                patient_info = analysis["patient_info"]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("👤 Personal Details")
                                    st.write(f"**Name:** {patient_info.get('name', 'N/A')}")
                                    st.write(f"**Age:** {patient_info.get('age', 'N/A')}")
                                    st.write(f"**Gender:** {patient_info.get('gender', 'N/A')}")
                                    st.write(f"**DOB:** {patient_info.get('date_of_birth', 'N/A')}")
                                    st.write(f"**Patient ID:** {patient_info.get('patient_id', 'N/A')}")
                                
                                with col2:
                                    st.subheader("📞 Contact Information")
                                    st.write(f"**Address:** {patient_info.get('address', 'N/A')}")
                                    st.write(f"**Phone:** {patient_info.get('phone_number', 'N/A')}")
                                    st.write(f"**Emergency Contact:** {patient_info.get('emergency_contact', 'N/A')}")
                                    st.write(f"**Insurance:** {patient_info.get('insurance_info', 'N/A')}")
                            
                            if analysis.get("visit_details"):
                                st.subheader("🏥 Visit Details")
                                visit = analysis["visit_details"]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Visit Date:** {visit.get('date_of_visit', 'N/A')}")
                                    st.write(f"**Visit Type:** {visit.get('visit_type', 'N/A')}")
                                    st.write(f"**Chief Complaint:** {visit.get('chief_complaint', 'N/A')}")
                                with col2:
                                    st.write(f"**Admission Date:** {visit.get('admission_date', 'N/A')}")
                                    st.write(f"**Discharge Date:** {visit.get('discharge_date', 'N/A')}")
                                    st.write(f"**Referring Physician:** {visit.get('referring_physician', 'N/A')}")
                            
                            # Vital Signs
                            if analysis.get("vital_signs"):
                                st.subheader("💓 Vital Signs")
                                vital_signs = analysis["vital_signs"]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🩸 Blood Pressure", 
                                             f"{vital_signs.get('blood_pressure_systolic', 'N/A')}/{vital_signs.get('blood_pressure_diastolic', 'N/A')}")
                                    st.metric("🫀 Heart Rate", f"{vital_signs.get('heart_rate', 'N/A')} bpm")
                                    st.metric("🌡️ Temperature", f"{vital_signs.get('temperature', 'N/A')}°F")
                                
                                with col2:
                                    st.metric("🫁 Respiratory Rate", f"{vital_signs.get('respiratory_rate', 'N/A')}/min")
                                    st.metric("🩸 O2 Saturation", f"{vital_signs.get('oxygen_saturation', 'N/A')}%")
                                    st.metric("⚖️ Weight", f"{vital_signs.get('weight', 'N/A')} lbs")
                                
                                with col3:
                                    st.metric("📏 Height", f"{vital_signs.get('height', 'N/A')} in")
                                    st.metric("📊 BMI", vital_signs.get('bmi', 'N/A'))
                                    st.metric("😣 Pain Scale", f"{vital_signs.get('pain_scale', 'N/A')}/10")
                        
                        with tab4:
                            st.header("💊 Medical Data")
                            
                            # Lab Results
                            if analysis.get("lab_results"):
                                st.subheader("🔬 Laboratory Results")
                                lab_results = analysis["lab_results"]
                                
                                if lab_results:
                                    # Create DataFrame for better display
                                    lab_data = []
                                    for result in lab_results:
                                        if isinstance(result, dict):
                                            lab_data.append({
                                                "Test Name": result.get("test_name", "Unknown"),
                                                "Value": result.get("value", "N/A"),
                                                "Normal Range": result.get("normal_range", "N/A"),
                                                "Status": result.get("status", "Unknown"),
                                                "Date": result.get("date", "N/A")
                                            })
                                    
                                    if lab_data:
                                        df_labs = pd.DataFrame(lab_data)
                                        st.dataframe(df_labs, use_container_width=True)
                                else:
                                    st.info("No laboratory results found")
                            
                            # Medications
                            if analysis.get("medications"):
                                st.subheader("💊 Current Medications")
                                medications = analysis["medications"]
                                
                                if medications:
                                    med_data = []
                                    for med in medications:
                                        if isinstance(med, dict):
                                            med_data.append({
                                                "Medication": med.get("name", "Unknown"),
                                                "Dosage": med.get("dosage", "N/A"),
                                                "Frequency": med.get("frequency", "N/A"),
                                                "Route": med.get("route", "N/A"),
                                                "Start Date": med.get("start_date", "N/A"),
                                                "Duration": med.get("duration", "N/A"),
                                                "Prescribing Doctor": med.get("prescribing_doctor", "N/A")
                                            })
                                    
                                    if med_data:
                                        df_meds = pd.DataFrame(med_data)
                                        st.dataframe(df_meds, use_container_width=True)
                                else:
                                    st.info("No medications found")
                            
                            # Procedures
                            if analysis.get("procedures"):
                                st.subheader("⚕️ Procedures")
                                procedures = analysis["procedures"]
                                
                                if procedures:
                                    for i, proc in enumerate(procedures, 1):
                                        if isinstance(proc, dict):
                                            with st.expander(f"Procedure {i}: {proc.get('name', 'Unknown')}"):
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.write(f"**Date:** {proc.get('date', 'N/A')}")
                                                    st.write(f"**Doctor:** {proc.get('doctor', 'N/A')}")
                                                with col2:
                                                    st.write(f"**Outcome:** {proc.get('outcome', 'N/A')}")
                                                    st.write(f"**Complications:** {proc.get('complications', 'None')}")
                            
                            # Diagnoses
                            if analysis.get("diagnoses"):
                                st.subheader("🩺 Diagnoses")
                                diagnoses = analysis["diagnoses"]
                                
                                for i, diagnosis in enumerate(diagnoses, 1):
                                    if isinstance(diagnosis, dict):
                                        st.write(f"**{i}. Primary Diagnosis:** {diagnosis.get('primary_diagnosis', 'N/A')}")
                                        if diagnosis.get('secondary_diagnoses'):
                                            st.write(f"   **Secondary:** {', '.join(diagnosis['secondary_diagnoses'])}")
                                        st.write(f"   **ICD Code:** {diagnosis.get('icd_code', 'N/A')}")
                                        st.write(f"   **Severity:** {diagnosis.get('severity', 'N/A')}")
                                        st.write(f"   **Date:** {diagnosis.get('diagnosis_date', 'N/A')}")
                                    else:
                                        st.write(f"**{i}.** {diagnosis}")
                            
                            # Imaging Studies
                            if analysis.get("imaging_studies"):
                                st.subheader("🔍 Imaging Studies")
                                imaging = analysis["imaging_studies"]
                                
                                for study in imaging:
                                    if isinstance(study, dict):
                                        with st.expander(f"{study.get('type', 'Unknown')} - {study.get('body_part', 'N/A')}"):
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.write(f"**Date:** {study.get('date', 'N/A')}")
                                                st.write(f"**Radiologist:** {study.get('radiologist', 'N/A')}")
                                            with col2:
                                                st.write(f"**Findings:** {study.get('findings', 'N/A')}")
                            
                            # Allergies
                            if analysis.get("allergies"):
                                st.subheader("🚨 Allergies")
                                allergies = analysis["allergies"]
                                for allergy in allergies:
                                    st.warning(f"⚠️ {allergy}")
                            
                            # Medical History
                            if analysis.get("medical_history"):
                                st.subheader("📋 Medical History")
                                history = analysis["medical_history"]
                                for item in history:
                                    st.write(f"• {item}")
                        
                        with tab5:
                            st.header("📋 Recommendations & Schedule")
                            
                            # Doctor Recommendations
                            if analysis.get("doctor_recommendations"):
                                st.subheader("👨‍⚕️ Doctor's Recommendations")
                                for rec in analysis["doctor_recommendations"]:
                                    st.info(f"💡 {rec}")
                            
                            # Discharge Instructions
                            if analysis.get("discharge_instructions"):
                                st.subheader("🏠 Discharge Instructions")
                                for instruction in analysis["discharge_instructions"]:
                                    st.write(f"📝 {instruction}")
                            
                            # Appointments Schedule
                            if analysis.get("appointments_schedule"):
                                st.subheader("📅 Upcoming Appointments")
                                appointments = analysis["appointments_schedule"]
                                
                                for appointment in appointments:
                                    if isinstance(appointment, dict):
                                        with st.container():
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.write(f"**Type:** {appointment.get('type', 'N/A')}")
                                                st.write(f"**Date:** {appointment.get('date', 'N/A')}")
                                            with col2:
                                                st.write(f"**Time:** {appointment.get('time', 'N/A')}")
                                                st.write(f"**Doctor:** {appointment.get('doctor', 'N/A')}")
                                            with col3:
                                                st.write(f"**Purpose:** {appointment.get('purpose', 'N/A')}")
                                                st.write(f"**Location:** {appointment.get('location', 'N/A')}")
                                            st.divider()
                            
                            # Key Findings
                            if analysis.get("key_findings"):
                                st.subheader("🔍 Key Medical Findings")
                                for finding in analysis["key_findings"]:
                                    st.success(f"✅ {finding}")
                            
                            # Risk Factors
                            if analysis.get("risk_factors"):
                                st.subheader("⚠️ Identified Risk Factors")
                                for risk in analysis["risk_factors"]:
                                    st.warning(f"⚠️ {risk}")
                            
                            # Follow-up Required
                            if analysis.get("follow_up_required", "N/A") != "N/A":
                                st.subheader("📅 Follow-up Required")
                                st.info(analysis["follow_up_required"])
                            
                            # Social History
                            if analysis.get("social_history"):
                                st.subheader("👥 Social History")
                                social = analysis["social_history"]
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Smoking:** {social.get('smoking', 'N/A')}")
                                    st.write(f"**Alcohol:** {social.get('alcohol', 'N/A')}")
                                with col2:
                                    st.write(f"**Occupation:** {social.get('occupation', 'N/A')}")
                                    st.write(f"**Exercise:** {social.get('exercise', 'N/A')}")
                            
                            # Billing Information
                            if analysis.get("billing_info"):
                                st.subheader("💰 Billing Information")
                                billing = analysis["billing_info"]
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Total Charges", billing.get("total_charges", "N/A"))
                                    st.metric("Insurance Coverage", billing.get("insurance_coverage", "N/A"))
                                with col2:
                                    st.metric("Patient Responsibility", billing.get("patient_responsibility", "N/A"))
                                    st.metric("Payment Status", billing.get("payment_status", "N/A"))
                        
                        with tab6:
                            st.header("📥 Download Comprehensive Medical Data")
                            
                            # Add metadata for download
                            download_data = {
                                "extraction_metadata": {
                                    "file_name": uploaded_file.name,
                                    "extraction_timestamp": datetime.now().isoformat(),
                                    "analyzer_version": "2.0",
                                    "total_data_points": sum([
                                        len(analysis.get("medications", [])),
                                        len(analysis.get("lab_results", [])),
                                        len(analysis.get("procedures", [])),
                                        len(analysis.get("diagnoses", [])),
                                        len(analysis.get("appointments_schedule", [])),
                                        len(analysis.get("risk_factors", []))
                                    ])
                                },
                                "medical_data": analysis,
                                "summary_statistics": stats
                            }
                            
                            # Format JSON for download
                            json_data = json.dumps(download_data, indent=2, ensure_ascii=False)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.download_button(
                                    label="📥 Download Complete Analysis (JSON)",
                                    data=json_data,
                                    file_name=f"medical_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json",
                                    help="Download the complete medical analysis with all extracted data"
                                )
                            
                            with col2:
                                # Create a summary report
                                summary_report = f"""
# Medical Analysis Summary Report
**File:** {uploaded_file.name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Medications: {stats['total_medications']}
- Lab Tests: {stats['total_lab_tests']}
- Procedures: {stats['total_procedures']}
- Diagnoses: {stats['total_diagnoses']}
- Appointments: {stats['total_appointments']}
- Risk Factors: {stats['risk_factor_count']}

## Patient Information
- Name: {analysis.get('patient_info', {}).get('name', 'N/A')}
- Age: {analysis.get('patient_info', {}).get('age', 'N/A')}
- Gender: {analysis.get('patient_info', {}).get('gender', 'N/A')}

## Key Findings
{chr(10).join(['- ' + str(finding) for finding in analysis.get('key_findings', ['No key findings recorded'])])}

## Recommendations
{chr(10).join(['- ' + str(rec) for rec in analysis.get('doctor_recommendations', ['No recommendations recorded'])])}
"""
                                
                                st.download_button(
                                    label="📄 Download Summary Report (TXT)",
                                    data=summary_report,
                                    file_name=f"medical_summary_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    help="Download a human-readable summary report"
                                )
                            
                            # Display preview of JSON structure
                            st.subheader("📋 Data Structure Preview")
                            st.json({
                                "extraction_metadata": download_data["extraction_metadata"],
                                "medical_data_keys": list(analysis.keys()),
                                "summary_statistics": stats
                            })
                            
                            # Data completeness indicator
                            st.subheader("📊 Data Completeness")
                            
                            completeness_data = []
                            sections = [
                                ("Administrative Info", analysis.get("administrative_info", {})),
                                ("Patient Info", analysis.get("patient_info", {})),
                                ("Vital Signs", analysis.get("vital_signs", {})),
                                ("Lab Results", analysis.get("lab_results", [])),
                                ("Medications", analysis.get("medications", [])),
                                ("Procedures", analysis.get("procedures", [])),
                                ("Diagnoses", analysis.get("diagnoses", []))
                            ]
                            
                            for section_name, section_data in sections:
                                if isinstance(section_data, dict):
                                    filled_fields = sum(1 for v in section_data.values() if v and v != "N/A")
                                    total_fields = len(section_data)
                                    completeness = (filled_fields / total_fields * 100) if total_fields > 0 else 0
                                else:
                                    completeness = 100 if section_data else 0
                                
                                completeness_data.append({
                                    "Section": section_name,
                                    "Completeness": completeness
                                })
                            
                            if completeness_data:
                                df_completeness = pd.DataFrame(completeness_data)
                                fig_completeness = px.bar(
                                    df_completeness, 
                                    x="Section", 
                                    y="Completeness",
                                    title="📊 Data Extraction Completeness by Section",
                                    color="Completeness",
                                    color_continuous_scale="RdYlGn"
                                )
                                fig_completeness.update_layout(height=400)
                                st.plotly_chart(fig_completeness, use_container_width=True)
                    
                    else:
                        st.error("❌ Failed to analyze the medical document. Please check the document quality and try again.")
            else:
                st.error("❌ Failed to extract text from the PDF. Please ensure the document is readable and try again.")

if __name__ == "__main__":
    main()