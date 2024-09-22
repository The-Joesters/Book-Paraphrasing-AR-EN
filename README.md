# Multilingual PDF Paraphrasing API

## Overview

This Flask-based web application provides an API endpoint to paraphrase text extracted from PDF files. It supports both Arabic and English languages. The application automatically detects the language of the uploaded PDF, processes the text accordingly, and returns the paraphrased content.

## Features

- **Language Detection**: Automatically detects if the PDF content is in Arabic or English.
- **Text Extraction**: Extracts text from PDF files using `pdfplumber`.
- **Text Cleaning**: Cleans the extracted text by removing URLs, numbers, special characters, and extra spaces.
- **Semantic Chunking**: Divides text into semantically coherent chunks for better paraphrasing.
- **Paraphrasing**:
  - **English**: Uses `t5-base` model for paraphrasing.
  - **Arabic**: Uses `google/mt5-base` model for paraphrasing.
- **Arabic Text Handling**: Corrects Arabic text direction and reshapes it for proper display.
- **RESTful API**: Provides a `/paraphrase` endpoint to upload PDFs and receive paraphrased text.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [API Endpoint](#api-endpoint)
- [Code Structure](#code-structure)
- [Models Used](#models-used)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Requirements

- Python 3.6 or higher
- CUDA-enabled GPU (recommended for performance)
- pip package manager

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/multilingual-pdf-paraphraser.git
   cd multilingual-pdf-paraphraser
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**

   ```python
   import nltk
   nltk.download('punkt')
   ```

5. **Download Stanza Models**

   ```python
   import stanza
   stanza.download('ar')
   ```

## Usage

### Running the Application

Start the Flask application:

```bash
python app.py
```

The application will run on `http://127.0.0.1:5000/` by default.

### API Endpoint

#### `POST /paraphrase`

- **Description**: Accepts a PDF file and returns the paraphrased text.
- **Content-Type**: `multipart/form-data`
- **Form Data**:
  - `file`: The PDF file to be paraphrased.

**Example Request using `curl`:**

```bash
curl -X POST -F 'file=@/path/to/your/file.pdf' http://127.0.0.1:5000/paraphrase
```

**Example Response:**

```json
{
  "summary": "Your paraphrased text here."
}
```

**Error Handling:**

- **400 Bad Request**: Missing file or invalid input.
- **500 Internal Server Error**: Error during processing.

## Code Structure

- **app.py**: Main application file containing the Flask app and all functions.
- **Functions**:
  - `extract_text_from_pdf(pdf_path)`: Extracts text from a PDF.
  - `fix_arabic_text(text)`: Fixes the direction and reshapes Arabic text.
  - `clean_text(text)`: Cleans English text.
  - `clean_arabic_text(text)`: Cleans Arabic text.
  - `divide_by_semantics_with_length(text, ...)`: Chunks English text semantically.
  - `chunk_arabic_text(text, ...)`: Chunks Arabic text.
  - `paraphrase_chunks_en(chunks, ...)`: Paraphrases English text chunks.
  - `paraphrase_chunks_ar(chunks, ...)`: Paraphrases Arabic text chunks.
  - `generate_txt(summary_text, ...)`: Generates a text file with the paraphrased content.
  - `paraphrase_english(book_text, ...)`: Pipeline for English paraphrasing.
  - `paraphrase_arabic(pdf_path, ...)`: Pipeline for Arabic paraphrasing.
  - `detect_language_and_paraphrase(pdf_path, ...)`: Detects language and triggers the appropriate pipeline.

## Models Used

- **English Paraphrasing**:
  - Model: `t5-base`
  - Library: `transformers`
- **Arabic Paraphrasing**:
  - Model: `google/mt5-base`
  - Library: `transformers`
- **Semantic Similarity**:
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Library: `sentence-transformers`

## Notes

- **Performance**: Using a GPU is highly recommended for handling large documents and improving paraphrasing speed.
- **Limitations**: Paraphrasing quality may vary depending on the complexity of the input text.
- **Customization**: Parameters like chunk size and paraphrasing models can be adjusted for better results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

