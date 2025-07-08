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
            # Estimate token count (rough: 1 token ≈ 4 chars)
            approx_tokens = len(text) // 4
            max_context_tokens = 7000  # Leave room for prompt and response
            max_chunk_chars = max_context_tokens * 4  # Approx chars per chunk

            # Base prompt template
            base_prompt = """
            Analyze the following medical document chunk and extract ALL available information in JSON format. 
            If any information is not found in the chunk, mark it as "N/A". Ensure the output is a valid JSON object.

            Medical Document Chunk:
            {text}

            Provide a comprehensive analysis in this JSON structure:
            {{
                "document_metadata": {{
                    "extraction_date": "{extraction_date}",
                    "document_type": "medical_report",
                    "file_source": "uploaded_pdf",
                    "analysis_confidence": "high/medium/low",
                    "text_length": {text_length},
                    "extraction_method": "AI_analysis"
                }},
                "administrative_info": {{
                    "bill_number": "N/A",
                    "mr_number": "N/A",
                    "room_ward_number": "N/A",
                    "hospital_name": "N/A",
                    "hospital_address": "N/A",
                    "hospital_phone": "N/A",
                    "department": "N/A",
                    "admission_number": "N/A"
                }},
                "patient_info": {{
                    "name": "N/A",
                    "age": "N/A",
                    "gender": "N/A",
                    "date_of_birth": "N/A",
                    "address": "N/A",
                    "phone_number": "N/A",
                    "emergency_contact": "N/A",
                    "insurance_info": "N/A",
                    "patient_id": "N/A"
                }},
                "visit_details": {{
                    "date_of_visit": "N/A",
                    "admission_date": "N/A",
                    "discharge_date": "N/A",
                    "visit_type": "N/A",
                    "chief_complaint": "N/A",
                    "referring_physician": "N/A"
                }},
                "medical_staff": {{
                    "attending_physician": "N/A",
                    "consultant_name": "N/A",
                    "resident_doctor": "N/A",
                    "nurse_in_charge": "N/A",
                    "other_staff": []
                }},
                "vital_signs": {{
                    "blood_pressure_systolic": "N/A",
                    "blood_pressure_diastolic": "N/A",
                    "heart_rate": "N/A",
                    "temperature": "N/A",
                    "respiratory_rate": "N/A",
                    "oxygen_saturation": "N/A",
                    "weight": "N/A",
                    "height": "N/A",
                    "bmi": "N/A",
                    "pain_scale": "N/A"
                }},
                "lab_results": [],
                "medications": [],
                "procedures": [],
                "diagnoses": [],
                "imaging_studies": [],
                "appointments_schedule": [],
                "doctor_recommendations": [],
                "discharge_instructions": [],
                "key_findings": [],
                "risk_factors": [],
                "allergies": [],
                "medical_history": [],
                "family_history": [],
                "social_history": {{
                    "smoking": "N/A",
                    "alcohol": "N/A",
                    "occupation": "N/A",
                    "exercise": "N/A"
                }},
                "follow_up_required": "N/A",
                "billing_info": {{
                    "total_charges": "N/A",
                    "insurance_coverage": "N/A",
                    "patient_responsibility": "N/A",
                    "payment_status": "N/A"
                }},
                "chart_data": {{
                    "trend_analysis": [],
                    "comparison_data": [],
                    "time_series": []
                }}
            }}
            
            Ensure the response is a valid JSON object enclosed in curly braces. Extract ALL numerical values for charts.
            """
            
            # Initialize result dictionary
            final_result = {}
            
            if approx_tokens <= max_context_tokens:
                # Process entire text if within context limit
                st.info(f"Analyzing document with {len(text)} characters (~{approx_tokens} tokens)...")
                with st.spinner("Analyzing with Llama 3 via OpenRouter..."):
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "meta-llama/llama-3.1-8b-instruct",
                        "messages": [
                            {
                                "role": "user",
                                "content": base_prompt.format(
                                    text=text,
                                    extraction_date=datetime.now().isoformat(),
                                    text_length=len(text)
                                )
                            }
                        ],
                        "max_tokens": 6000,  # Increased to handle larger responses
                        "temperature": 0.1
                    }
                    
                    try:
                        response = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=30
                        )
                        response.raise_for_status()
                        
                        analysis_text = response.json()["choices"][0]["message"]["content"]
                        # Log raw response for debugging
                        with open(f"api_response_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w", encoding="utf-8") as f:
                            f.write(analysis_text)
                        
                        # Improved regex to handle JSON within markdown or plain text
                        json_match = re.search(r'\{[\s\S]*\}', analysis_text)
                        if not json_match:
                            st.error(f"Failed to extract valid JSON from API response. Raw response saved to api_response_single.txt")
                            st.text_area("Raw API Response", analysis_text[:2000], height=200)
                            return {}
                        
                        try:
                            final_result = json.loads(json_match.group())
                        except json.JSONDecodeError as e:
                            st.error(f"JSON parsing error: {str(e)}. Raw response saved to api_response_single.txt")
                            st.text_area("Raw API Response", analysis_text[:2000], height=200)
                            return {}
                    except requests.exceptions.RequestException as e:
                        st.error(f"API request failed: {str(e)}")
                        return {}
            else:
                # Chunk the text at paragraph breaks to preserve context
                st.warning(f"Document is large ({len(text)} chars, ~{approx_tokens} tokens). Processing in chunks...")
                chunks = []
                current_chunk = ""
                for paragraph in text.split("\n\n"):
                    if len(current_chunk) + len(paragraph) + 2 <= max_chunk_chars:
                        current_chunk += paragraph + "\n\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = paragraph + "\n\n"
                if current_chunk:
                    chunks.append(current_chunk)
                
                st.info(f"Processing {len(chunks)} chunks...")
                progress_bar = st.progress(0)
                chunk_results = []
                
                for i, chunk in enumerate(chunks):
                    with st.spinner(f"Analyzing chunk {i+1}/{len(chunks)}..."):
                        headers = {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "model": "meta-llama/llama-3.1-8b-instruct",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": base_prompt.format(
                                        text=chunk,
                                        extraction_date=datetime.now().isoformat(),
                                        text_length=len(chunk)
                                    )
                                }
                            ],
                            "max_tokens": 6000,
                            "temperature": 0.1
                        }
                        
                        try:
                            response = requests.post(
                                "https://openrouter.ai/api/v1/chat/completions",
                                headers=headers,
                                json=payload,
                                timeout=30
                            )
                            response.raise_for_status()
                            
                            analysis_text = response.json()["choices"][0]["message"]["content"]
                            # Log raw response for debugging
                            with open(f"api_response_chunk_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w", encoding="utf-8") as f:
                                f.write(analysis_text)
                            
                            # Improved regex to handle JSON within markdown or plain text
                            json_match = re.search(r'\{[\s\S]*\}', analysis_text)
                            if not json_match:
                                st.warning(f"Failed to extract valid JSON from chunk {i+1}. Skipping chunk. Raw response saved to api_response_chunk_{i+1}.txt")
                                st.text_area(f"Raw API Response for Chunk {i+1}", analysis_text[:2000], height=200)
                                continue
                            
                            try:
                                chunk_result = json.loads(json_match.group())
                                chunk_results.append(chunk_result)
                            except json.JSONDecodeError as e:
                                st.warning(f"JSON parsing error in chunk {i+1}: {str(e)}. Skipping chunk. Raw response saved to api_response_chunk_{i+1}.txt")
                                st.text_area(f"Raw API Response for Chunk {i+1}", analysis_text[:2000], height=200)
                                continue
                        except requests.exceptions.RequestException as e:
                            st.warning(f"API request failed for chunk {i+1}: {str(e)}. Skipping chunk.")
                            continue
                        
                    progress_bar.progress((i + 1) / len(chunks))
                
                progress_bar.empty()
                
                if not chunk_results:
                    st.error("No valid chunks processed. Please try a smaller document or check API settings.")
                    return {}
                
                # Merge chunk results
                final_result = self.merge_chunk_results(chunk_results)
            
            return final_result
                
        except Exception as e:
            st.error(f"Unexpected error analyzing medical data: {str(e)}")
            return {}
    
    def merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple JSON results from chunked text analysis"""
        if not chunk_results:
            return {}
        
        # Initialize merged result with the first chunk's structure
        merged_result = chunk_results[0].copy()
        
        # Update document_metadata
        merged_result["document_metadata"]["text_length"] = sum(
            chunk["document_metadata"]["text_length"] for chunk in chunk_results
        )
        merged_result["document_metadata"]["extraction_date"] = datetime.now().isoformat()
        
        # Merge list-based fields
        list_fields = [
            "lab_results", "medications", "procedures", "diagnoses", "imaging_studies",
            "appointments_schedule", "doctor_recommendations", "discharge_instructions",
            "key_findings", "risk_factors", "allergies", "medical_history", "family_history"
        ]
        
        for field in list_fields:
            merged_result[field] = []
            for chunk in chunk_results:
                if chunk.get(field):
                    merged_result[field].extend(chunk[field])
        
        # For dictionary fields, take the first non-empty value or keep N/A
        dict_fields = [
            "administrative_info", "patient_info", "visit_details", "medical_staff",
            "vital_signs", "social_history", "billing_info", "chart_data"
        ]
        
        for field in dict_fields:
            for chunk in chunk_results:
                if chunk.get(field) and any(v != "N/A" and v for v in chunk[field].values()):
                    merged_result[field] = chunk[field]
                    break
        
        # Update analysis_confidence based on the lowest confidence across chunks
        confidences = {"high": 3, "medium": 2, "low": 1}
        min_confidence = min(
            confidences.get(chunk["document_metadata"]["analysis_confidence"], 1)
            for chunk in chunk_results
        )
        merged_result["document_metadata"]["analysis_confidence"] = next(
            k for k, v in confidences.items() if v == min_confidence
        )
        
        return merged_result
    
    def display_metadata_section(self, title: str, data: Dict[str, Any], icon: str = "📋"):
        """Display a metadata section with proper formatting"""
        st.subheader(f"{icon} {title}")
        
        if not data:
            st.warning("No data available for this section")
            return
        
        # Create a formatted display
        with st.container():
            # Use columns for better layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Field**")
            with col2:
                st.markdown("**Value**")
            
            st.divider()
            
            for key, value in data.items():
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Format field name
                    field_name = key.replace('_', ' ').title()
                    st.markdown(f"**{field_name}:**")
                
                with col2:
                    # Format value based on type
                    if isinstance(value, list):
                        if value:
                            for item in value:
                                st.write(f"• {item}")
                        else:
                            st.write("*No items*")
                    elif isinstance(value, dict):
                        st.json(value)
                    else:
                        # Handle N/A values
                        if str(value) == "N/A":
                            st.write("*Not Available*")
                        else:
                            st.write(str(value))
    
    def display_list_metadata_table(self, title: str, data: List[Dict[str, Any]], icon: str = "📋"):
        """Display list-based metadata in a tabular format"""
        st.subheader(f"{icon} {title}")
        
        if not data:
            st.warning(f"No {title.lower()} data available")
            return
        
        # Display count
        st.info(f"Total {title}: {len(data)}")
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Replace 'N/A' with a more readable format for display
        df = df.fillna("Not Available")
        
        # Format column names
        df.columns = [col.replace('_', ' ').title() for col in df.columns]
        
        # Display the table
        st.dataframe(df, use_container_width=True)
    
    def display_simple_list_table(self, title: str, data: List[str], icon: str = "📋"):
        """Display simple list data (e.g., recommendations, instructions) in a single-column table"""
        st.subheader(f"{icon} {title}")
        
        if not data:
            st.warning(f"No {title.lower()} data available")
            return
        
        # Display count
        st.info(f"Total {title}: {len(data)}")
        
        # Convert to DataFrame with a single column
        df = pd.DataFrame(data, columns=[title[:-1]])  # Remove 's' for singular column name
        
        # Display the table
        st.dataframe(df, use_container_width=True)
    
    def get_item_identifier(self, item: Dict[str, Any]) -> str:
        """Get an identifier for display purposes"""
        if isinstance(item, dict):
            # Try common identifier fields
            for field in ['name', 'test_name', 'type', 'primary_diagnosis', 'date']:
                if field in item and item[field] and item[field] != "N/A":
                    return str(item[field])
            # Return first non-N/A value
            for key, value in item.items():
                if value and str(value) != "N/A":
                    return str(value)
        return "Details"
    
    def create_metadata_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive metadata summary"""
        summary = {
            "extraction_timestamp": datetime.now().isoformat(),
            "document_analysis_stats": {
                "total_sections": len(analysis_data),
                "populated_sections": sum(1 for section in analysis_data.values() if section and section != "N/A"),
                "data_completeness_percentage": 0
            },
            "content_statistics": {
                "medications_count": len(analysis_data.get("medications", [])),
                "lab_results_count": len(analysis_data.get("lab_results", [])),
                "procedures_count": len(analysis_data.get("procedures", [])),
                "diagnoses_count": len(analysis_data.get("diagnoses", [])),
                "imaging_studies_count": len(analysis_data.get("imaging_studies", [])),
                "appointments_count": len(analysis_data.get("appointments_schedule", [])),
                "risk_factors_count": len(analysis_data.get("risk_factors", [])),
                "allergies_count": len(analysis_data.get("allergies", [])),
                "recommendations_count": len(analysis_data.get("doctor_recommendations", [])),
                "discharge_instructions_count": len(analysis_data.get("discharge_instructions", [])),
                "billing_info_completeness": self.calculate_completeness(analysis_data.get("billing_info", {}))
            }
        }
        
        # Calculate overall completeness
        total_completeness = sum([
            summary["content_statistics"]["billing_info_completeness"],
            self.calculate_completeness(analysis_data.get("vital_signs", {})),
            self.calculate_completeness(analysis_data.get("patient_info", {})),
            self.calculate_completeness(analysis_data.get("administrative_info", {})),
            self.calculate_completeness(analysis_data.get("visit_details", {}))
        ])
        summary["document_analysis_stats"]["data_completeness_percentage"] = total_completeness / 5 if total_completeness > 0 else 0
        
        return summary
    
    def calculate_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate completeness percentage for a data section"""
        if not data:
            return 0.0
        
        total_fields = len(data)
        populated_fields = sum(1 for value in data.values() if value and str(value) != "N/A")
        
        return (populated_fields / total_fields * 100) if total_fields > 0 else 0.0
    
    def display_complete_metadata(self, analysis_data: Dict[str, Any]):
        """Display all extracted information in metadata format"""
        st.header("🔍 Complete Medical Document Metadata")
        
        # Create metadata summary
        metadata_summary = self.create_metadata_summary(analysis_data)
        
        # Display metadata summary first
        st.subheader("📊 Metadata Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sections", metadata_summary["document_analysis_stats"]["total_sections"])
        with col2:
            st.metric("Populated Sections", metadata_summary["document_analysis_stats"]["populated_sections"])
        with col3:
            st.metric("Data Completeness", f"{metadata_summary['document_analysis_stats']['data_completeness_percentage']:.1f}%")
        with col4:
            st.metric("Content Items", sum([v for k, v in metadata_summary["content_statistics"].items() if k != "billing_info_completeness"]))
        
        # Display detailed metadata in organized sections
        metadata_tabs = st.tabs([
            "📋 Document Metadata",
            "🏥 Administrative Data", 
            "👤 Patient Information",
            "🩺 Medical Data",
            "💰 Billing",
            "🔍 Raw Data Structure"
        ])
        
        with metadata_tabs[0]:
            st.header("📋 Document Metadata")
            
            # Document metadata
            if analysis_data.get("document_metadata"):
                self.display_metadata_section("Document Information", analysis_data["document_metadata"], "📄")
            
            # Visit details
            if analysis_data.get("visit_details"):
                self.display_metadata_section("Visit Details", analysis_data["visit_details"], "🏥")
            
            # Medical staff
            if analysis_data.get("medical_staff"):
                self.display_metadata_section("Medical Staff", analysis_data["medical_staff"], "👨‍⚕️")
        
        with metadata_tabs[1]:
            st.header("🏥 Administrative Data")
            
            # Administrative info
            if analysis_data.get("administrative_info"):
                self.display_metadata_section("Administrative Information", analysis_data["administrative_info"], "📋")
        
        with metadata_tabs[2]:
            st.header("👤 Patient Information")
            
            # Patient info
            if analysis_data.get("patient_info"):
                self.display_metadata_section("Patient Details", analysis_data["patient_info"], "👤")
            
            # Vital signs
            if analysis_data.get("vital_signs"):
                self.display_metadata_section("Vital Signs", analysis_data["vital_signs"], "💓")
            
            # Social history
            if analysis_data.get("social_history"):
                self.display_metadata_section("Social History", analysis_data["social_history"], "👥")
            
            # Medical history
            if analysis_data.get("medical_history"):
                st.subheader("📋 Medical History")
                for item in analysis_data["medical_history"]:
                    st.write(f"• {item}")
            
            # Family history
            if analysis_data.get("family_history"):
                st.subheader("👨‍👩‍👧‍👦 Family History")
                for item in analysis_data["family_history"]:
                    st.write(f"• {item}")
            
            # Allergies
            if analysis_data.get("allergies"):
                st.subheader("🚨 Allergies")
                for allergy in analysis_data["allergies"]:
                    st.warning(f"⚠️ {allergy}")
        
        with metadata_tabs[3]:
            st.header("🩺 Medical Data")
            
            # Lab results
            if analysis_data.get("lab_results"):
                self.display_list_metadata_table("Lab Results", analysis_data["lab_results"], "🔬")
            
            # Medications
            if analysis_data.get("medications"):
                self.display_list_metadata_table("Medications", analysis_data["medications"], "💊")
            
            # Procedures
            if analysis_data.get("procedures"):
                self.display_list_metadata_table("Procedures", analysis_data["procedures"], "⚕️")
            
            # Diagnoses
            if analysis_data.get("diagnoses"):
                self.display_list_metadata_table("Diagnoses", analysis_data["diagnoses"], "🩺")
            
            # Imaging studies
            if analysis_data.get("imaging_studies"):
                self.display_list_metadata_table("Imaging Studies", analysis_data["imaging_studies"], "🔍")
            
            # Appointments
            if analysis_data.get("appointments_schedule"):
                self.display_list_metadata_table("Appointments", analysis_data["appointments_schedule"], "📅")
            
            # Recommendations
            if analysis_data.get("doctor_recommendations"):
                self.display_simple_list_table("Doctor Recommendations", analysis_data["doctor_recommendations"], "👨‍⚕️")
            
            # Discharge instructions
            if analysis_data.get("discharge_instructions"):
                self.display_simple_list_table("Discharge Instructions", analysis_data["discharge_instructions"], "🏠")
            
            # Key findings
            if analysis_data.get("key_findings"):
                self.display_simple_list_table("Key Findings", analysis_data["key_findings"], "🔍")
            
            # Risk factors
            if analysis_data.get("risk_factors"):
                self.display_simple_list_table("Risk Factors", analysis_data["risk_factors"], "⚠️")
        
        with metadata_tabs[4]:
            st.header("💰 Billing")
            
            # Billing info
            if analysis_data.get("billing_info"):
                self.display_metadata_section("Billing Information", analysis_data["billing_info"], "💰")
        
        with metadata_tabs[5]:
            st.header("🔍 Raw Data Structure")
            
            # Show complete raw data structure
            st.subheader("📊 Complete Extracted Data")
            st.json(analysis_data)
            
            # Data export section
            st.subheader("📥 Export Options")
            
            # Prepare complete metadata for download
            complete_metadata = {
                "metadata_summary": metadata_summary,
                "extracted_data": analysis_data,
                "export_timestamp": datetime.now().isoformat(),
                "export_format": "complete_metadata_json"
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download Complete Metadata (JSON)",
                    data=json.dumps(complete_metadata, indent=2, ensure_ascii=False),
                    file_name=f"complete_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Create structured metadata report
                metadata_report = self.create_metadata_report(metadata_summary, analysis_data)
                st.download_button(
                    label="📄 Download Metadata Report (TXT)",
                    data=metadata_report,
                    file_name=f"metadata_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

    def create_metadata_report(self, metadata_summary: Dict[str, Any], analysis_data: Dict[str, Any]) -> str:
        """Create a comprehensive metadata report"""
        report = f"""
# COMPREHENSIVE MEDICAL DOCUMENT METADATA REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## METADATA SUMMARY
Total Sections: {metadata_summary['document_analysis_stats']['total_sections']}
Populated Sections: {metadata_summary['document_analysis_stats']['populated_sections']}
Data Completeness: {metadata_summary['document_analysis_stats']['data_completeness_percentage']:.1f}%

## CONTENT STATISTICS
"""
        
        for key, value in metadata_summary["content_statistics"].items():
            if key != "billing_info_completeness":
                report += f"{key.replace('_', ' ').title()}: {value}\n"
        
        report += "\n## BILLING INFORMATION\n"
        billing_info = analysis_data.get("billing_info", {})
        if billing_info:
            for key, value in billing_info.items():
                report += f"{key.replace('_', ' ').title()}: {value}\n"
        else:
            report += "No billing information available\n"
        
        report += "\n## DETAILED SECTION BREAKDOWN\n"
        
        for section_name, section_data in analysis_data.items():
            if section_name in ["document_metadata", "billing_info"]:
                continue
            if section_data and section_data != "N/A":
                report += f"\n### {section_name.replace('_', ' ').upper()}\n"
                
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        report += f"  {key.replace('_', ' ').title()}: {value}\n"
                elif isinstance(section_data, list):
                    report += f"  Items: {len(section_data)}\n"
                    for i, item in enumerate(section_data[:3], 1):  # Show first 3 items
                        report += f"    {i}. {item}\n"
                    if len(section_data) > 3:
                        report += f"    ... and {len(section_data) - 3} more items\n"
                else:
                    report += f"  Value: {section_data}\n"
        
        return report

def main():
    st.set_page_config(
        page_title="Medical PDF Metadata Analyzer",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Medical PDF Metadata Analyzer")
    st.markdown("📤 Upload medical PDFs to extract and display comprehensive metadata with AI-powered analysis")
    
    # Initialize analyzer
    analyzer = MedicalPDFAnalyzer()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Check Tesseract configuration
        try:
            version = pytesseract.get_tesseract_version()
            st.success(f"✅ Tesseract OCR: {version}")
        except:
            st.error("❌ Tesseract not found. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        
        # Check API configuration
        if analyzer.model_loaded:
            st.success("✅ OpenRouter API configured")
        else:
            st.error("❌ OpenRouter API not configured")
            
        st.divider()
        st.subheader("Upload Settings")
        max_file_size = st.slider("Maximum file size (MB)", 1, 100, 50)
        enable_ocr = st.checkbox("Enable OCR for scanned PDFs", value=True)
        st.info("OCR is recommended for scanned documents but may increase processing time")
        
        st.divider()
        st.subheader("Analysis Settings")
        confidence_threshold = st.selectbox("Analysis confidence threshold", ["High", "Medium", "Low"], index=1)
        st.markdown("Select the confidence level for more reliable results")
        
        st.divider()
        st.subheader("About")
        st.markdown("Medical PDF Metadata Analyzer v1.0\nPowered by xAI's Grok and OpenRouter")
        st.markdown("[Source Code](https://github.com/your-org/medical-pdf-analyzer)")
    
    # File uploader
    st.header("📤 Upload Medical PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=False)
    
    if uploaded_file:
        # Validate file size
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > max_file_size:
            st.error(f"File size ({file_size_mb:.2f}MB) exceeds maximum limit of {max_file_size}MB")
        else:
            st.success(f"Uploaded: {uploaded_file.name} ({file_size_mb:.2f}MB)")
            
            # Extract text
            with st.spinner("Extracting text from PDF..."):
                text = analyzer.extract_text_from_pdf(uploaded_file)
            
            if text:
                st.success("Text extraction completed!")
                st.subheader("Extracted Text Preview")
                with st.expander("View raw extracted text"):
                    st.text_area("Extracted Text", text[:2000], height=300)  # Limit preview to 2000 chars
                
                # Analyze medical data
                with st.spinner(f"Analyzing medical data with {confidence_threshold.lower()} confidence..."):
                    analysis_data = analyzer.analyze_medical_data(text)
                
                if analysis_data:
                    st.success("Medical data analysis completed!")
                    # Display complete metadata
                    analyzer.display_complete_metadata(analysis_data)
                else:
                    st.error("Failed to analyze medical data. Please try another PDF or check API settings.")
            else:
                st.error("No text could be extracted from the PDF. Ensure it's a valid medical document.")
    
    # Footer
    st.divider()
    st.markdown("**Note**: This tool is for informational purposes only and should not be used for medical diagnosis or treatment decisions.")
    st.markdown("Developed with ❤️ by xAI | © 2025")

if __name__ == "__main__":
    main()