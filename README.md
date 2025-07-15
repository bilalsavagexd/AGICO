# 🏥 Medical PDF Metadata Analyzer

A Streamlit-based application that extracts and analyzes medical PDF documents using AI-powered text extraction and OCR capabilities.

## Features

- 📄 PDF text extraction using PyPDF2
- 🖼️ OCR processing for scanned documents using Tesseract
- 🤖 AI-powered medical data analysis using OpenRouter API
- 📊 Comprehensive metadata extraction and visualization
- 🔍 Interactive web interface built with Streamlit

## Quick Start with Docker

### Prerequisites

- Docker installed on your system
- Docker Compose installed on your system

### Option 1: Using Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AGICO
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

3. **Build and run the application**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   Open your browser and navigate to: `http://localhost:8501`

### Option 2: Using Docker directly

1. **Build the Docker image**
   ```bash
   docker build -t medical-pdf-analyzer .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 --env-file .env medical-pdf-analyzer
   ```

## Manual Installation (Alternative)

If you prefer to run without Docker:

### Prerequisites

- Python 3.11 or higher
- Tesseract OCR installed on your system
- Poppler utilities installed on your system

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AGICO
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies**
   
   **On Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install tesseract-ocr poppler-utils
   ```
   
   **On macOS:**
   ```bash
   brew install tesseract poppler
   ```
   
   **On Windows:**
   - Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
   - Download and install Poppler from: https://poppler.freedesktop.org/

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenRouter API key.

6. **Run the application**
   ```bash
   streamlit run main.py
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### OpenRouter API Setup

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Generate an API key
3. Add the key to your `.env` file

## Usage

1. **Start the application** using one of the methods above
2. **Upload a PDF file** using the file uploader
3. **Wait for processing** - the app will extract text and analyze the medical data
4. **View results** - comprehensive metadata and analysis will be displayed

## Docker Commands Reference

### Development Commands

```bash
# Build the image
docker-compose build

# Run in development mode (with volume mounting)
docker-compose up

# Run in detached mode
docker-compose up -d

# Stop the application
docker-compose down

# View logs
docker-compose logs

# Rebuild and run
docker-compose up --build

# Clean up containers and images
docker-compose down --rmi all
```

### Production Commands

```bash
# Build for production
docker build -t medical-pdf-analyzer:latest .

# Run in production mode
docker run -d -p 8501:8501 --env-file .env --name medical-analyzer medical-pdf-analyzer:latest

# Stop production container
docker stop medical-analyzer

# Remove production container
docker rm medical-analyzer
```

## Project Structure

```
AGICO/
├── main.py              # Main Streamlit application
├── requirements.txt     # Python dependencies
├── packages.txt        # System packages for deployment
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── .env               # Environment variables (create from .env.example)
├── .dockerignore      # Docker ignore file
└── README.md          # This file
```

## Dependencies

### Python Packages
- streamlit>=1.28.0
- PyPDF2>=3.0.1
- pdf2image>=1.16.3
- Pillow>=10.0.0
- pytesseract>=0.3.10
- plotly>=5.15.0
- pandas>=2.0.0
- python-dotenv>=1.0.0
- requests>=2.31.0

### System Dependencies
- tesseract-ocr (for OCR processing)
- poppler-utils (for PDF to image conversion)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub.

## Disclaimer

This tool is for informational purposes only and should not be used for medical diagnosis or treatment decisions.

---

**Developed with ❤️ by xAI | © 2025**
