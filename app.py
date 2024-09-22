from flask import Flask, request, jsonify, send_file
import os
import re
import nltk
from langdetect import detect
from fpdf import FPDF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import pdfplumber
import arabic_reshaper
from bidi.algorithm import get_display
from sentence_transformers import SentenceTransformer, util
import stanza
import torch
import tempfile
from werkzeug.utils import secure_filename


# Initialize Flask app
app = Flask(__name__)

# Load models and pipelines
nltk.download('punkt')
stanza.download('ar')

# Load pre-trained Sentence-BERT model for semantic embeddings (ensure GPU usage)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2',device=0)
paraphraser_en = pipeline("text2text-generation", model="t5-base",device=0)
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
paraphraser_ar = pipeline("text2text-generation", model=model, tokenizer=tokenizer,device=0)


# Initialize Stanza pipeline for Arabic
nlp_ar = stanza.Pipeline('ar', processors='tokenize')




# Extract text from the PDF file using pdfplumber
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join(page.extract_text() for page in pdf.pages)
    return text

# Function: Arabic Text Reshaping and Bidi Fix
def fix_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# Clean the text by removing URLs, numbers, and extra spaces
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\b[A-Za-z]\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_arabic_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\b[ء-ي]\b', '', text)
    text = re.sub(r'\b[A-Za-z]\b', '', text)
    text = re.sub(r'[?!؟"()«»:\-]', '', text)
    text = re.sub(r'\s+([,.،])', r'\1', text)
    text = re.sub(r'([,.،])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Clean and summarize all chunks
def clean_chunks(chunks):
    return [clean_arabic_text(chunk) for chunk in chunks]


"""## Step 6: Define the Function for Semantic Chunking in English"""

def divide_by_semantics_with_length(text, threshold=0.6, max_words=400, min_words=300):
    sentences = nltk.sent_tokenize(text)  # Use NLTK for sentence tokenization
    embeddings = sbert_model.encode(sentences, convert_to_tensor=True)
    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):
        similarity = util.pytorch_cos_sim(embeddings[i], embeddings[i-1])
        current_word_count = len(current_chunk.split())

        # If the next sentence makes the chunk exceed the max word limit
        if current_word_count + len(sentences[i].split()) > max_words:
            # Ensure the current chunk has at least min_words before breaking
            if current_word_count >= min_words:
                chunks.append(current_chunk.strip())  # Finalize the current chunk
                current_chunk = sentences[i]  # Start a new chunk
            else:
                # If the chunk is below min_words, add the sentence even if it exceeds max_words
                current_chunk += ' ' + sentences[i]
        elif similarity < threshold:
            # Break chunk if semantic similarity is low and the chunk meets the minimum word count
            if current_word_count >= min_words:
                chunks.append(current_chunk.strip())  # Finalize the current chunk
                current_chunk = sentences[i]  # Start a new chunk
            else:
                # If the chunk is too small, continue adding sentences
                current_chunk += ' ' + sentences[i]
        else:
            # Continue adding sentences to the current chunk
            current_chunk += ' ' + sentences[i]

    # Append the last chunk if it satisfies the minimum word condition
    if len(current_chunk.split()) >= min_words:
        chunks.append(current_chunk.strip())

    return chunks

"""## Step 7: Define the Function for Semantic Chunking in Arabic"""

def chunk_arabic_text(text, tokenizer, max_tokens=500):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ''
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                # إذا كانت الجملة نفسها تتجاوز الحد الأقصى، نقسمها إلى كلمات
                words = sentence.split()
                sub_chunk = ''
                sub_tokens = 0
                for word in words:
                    word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
                    if sub_tokens + word_tokens > max_tokens:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                            sub_chunk = word
                            sub_tokens = word_tokens
                        else:
                            sub_chunk = ''
                            sub_tokens = 0
                    else:
                        sub_chunk += ' ' + word
                        sub_tokens += word_tokens
                if sub_chunk:
                    chunks.append(sub_chunk.strip())
        else:
            current_chunk += ' ' + sentence
            current_tokens += sentence_tokens

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

"""## Step 8: Define the Paraphrasing and PDF Generation Functions

"""

# Paraphrase text chunks
def paraphrase_chunks_en(chunks, min_words=350, max_words=400, num_return_sequences=1):
    paraphrased_chunks = []
    for chunk in chunks:
        chunk_length = len(chunk.split())  # Get the word count of the original chunk

        try:
            # Use the paraphraser to generate paraphrases
            paraphrases = paraphraser_en(chunk, max_length=chunk_length, num_return_sequences=num_return_sequences, do_sample=False)
            paraphrased_text = paraphrases[0]['generated_text']  # Extract the paraphrased text

            # Ensure that the paraphrased text has between min_words and max_words
            paraphrased_words = paraphrased_text.split()
            if len(paraphrased_words) > max_words:
                paraphrased_text = ' '.join(paraphrased_words[:max_words])  # Trim if longer
            elif len(paraphrased_words) < min_words:
                paraphrased_text = paraphrased_text + ' ' + ' '.join(paraphrased_words[:min_words-len(paraphrased_words)])  # Repeat words if shorter

            paraphrased_chunks.append(paraphrased_text)
        except Exception as e:
            #print(f"Error paraphrasing chunk: {e}")
            paraphrased_chunks.append(chunk)  # Append the original chunk if paraphrasing fails

    return paraphrased_chunks


def paraphrase_chunks_ar(chunks, min_words=350, max_words=400, num_return_sequences=1):

    paraphrased_chunks = []
    for chunk in chunks:
        chunk_length = len(chunk.split())  # Get the word count of the original chunk

        try:
            paraphrases = paraphraser_ar(chunk, max_length=chunk_length, num_return_sequences=num_return_sequences, do_sample=False)
            paraphrased_text = paraphrases[0]['generated_text']  # Extract the paraphrased text

            # Ensure that the paraphrased text has between min_words and max_words
            paraphrased_words = paraphrased_text.split()
            if len(paraphrased_words) > max_words:
                paraphrased_text = ' '.join(paraphrased_words[:max_words])  # Trim if longer
            elif len(paraphrased_words) < min_words:
                paraphrased_text = paraphrased_text + ' ' + ' '.join(paraphrased_words[:min_words-len(paraphrased_words)])  # Repeat words if shorter

            paraphrased_chunks.append(paraphrased_text)
        except Exception as e:
            #print(f"Error paraphrasing chunk: {e}")
            paraphrased_chunks.append(chunk)  # Append the original chunk if paraphrasing fails

    return paraphrased_chunks


# Process the title based on the language
def get_title(language):
    if language == 'ar':
        title = "إعادة صباغة الكتاب"
    else:
        title = 'Book Paraphrased'
    return title


# Generate the text file
def generate_txt(summary_text, txt_output_path, language='en'):
    # Process the title
    title = get_title(language)

    # Process the body text
    if language == 'ar':
        reshaped_text = arabic_reshaper.reshape(summary_text)
        body_text = get_display(reshaped_text)
    else:
        body_text = summary_text

    # Define A4 page parameters
    characters_per_line = 80  # تقديريًا لعرض السطر في A4
    effective_line_width = characters_per_line

    # Adjust alignment based on language
    if language == 'ar':
        # For Arabic, define a function to right-align text
        def align_line(line):
            return line.rjust(effective_line_width)
    else:
        # For English, define a function to left-align text
        def align_line(line):
            return line.ljust(effective_line_width)

    # Center the title considering alignment
    centered_title = title.center(effective_line_width)

    # Format the body text with alignment
    formatted_body = ''
    for paragraph in body_text.split('\n'):
        words = paragraph.split()
        line = ''
        for word in words:
            if len(line) + len(word) + 1 <= effective_line_width:
                line += word + ' '
            else:
                # Strip extra space and align the line
                line = line.strip()
                formatted_line = align_line(line)
                formatted_body += formatted_line + '\n'
                line = word + ' '
        if line:
            line = line.strip()
            formatted_line = align_line(line)
            formatted_body += formatted_line + '\n'
        formatted_body += '\n'  # إضافة سطر فارغ بين الفقرات

    # Write the title and body to a text file
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write(centered_title + '\n\n')
        f.write(formatted_body)

"""## Step 9: Define the Paraphrasing Pipelines for English and Arabic

### **English Paraphrasing  Pipeline**
"""

def paraphrase_english(book_text, text_output_path):
    # Step 1: Divide text into semantic chunks
    semantic_chunks = divide_by_semantics_with_length(book_text)

    # Step 2: Clean the chunks
    cleaned_chunks = [clean_text(chunk) for chunk in semantic_chunks]

    # Step 3: Paraphrase the chunks
    paraphrased_chunks = paraphrase_chunks_en(cleaned_chunks)

    # Step 4: Generate text
    final_paraphrase = '\n\n'.join(paraphrased_chunks)
    generate_txt(final_paraphrase, text_output_path, language='en')

    print(f"Paraphrasing completed! Saved to {text_output_path}")

    return final_paraphrase

"""### **Arabic Paraphrasing Pipeline**

"""

def paraphrase_arabic(pdf_path, text_output_path):
    # Step 1: Extract text from PDF and fix Arabic text direction
    text = extract_text_from_pdf(pdf_path)
    fixed_text = fix_arabic_text(text)  # Fixing the text direction

    # Step 2: Chunk the text semantically
    chunks = chunk_arabic_text(fixed_text, tokenizer, max_tokens=400)

    # Step 3: Paraphrase the chunks
    paraphrased_chunks = paraphrase_chunks_ar(chunks)

    # Step 4: Clean the paraphrased chunks using the custom Arabic cleaning function
    cleaned_paraphrase = clean_chunks(paraphrased_chunks)

    # Step 5: Join the cleaned chunks into the final paraphrased text
    final_paraphrase = '\n\n'.join(cleaned_paraphrase)

    # Step 6: Fix the Arabic text direction before generating the text
    final_paraphrase_arabic = fix_arabic_text(final_paraphrase)
    generate_txt(final_paraphrase_arabic, text_output_path, language='ar')

    # Notify the user that the text has been created
    print(f"Paraphrasing completed! Saved to {text_output_path}")

    return final_paraphrase_arabic


"""## Step 10: Language Detection and Pipeline Execution

"""

def detect_language_and_paraphrase(pdf_path, text_output_path_ar="arabic_paraphrase.txt", text_output_path_en="english_paraphrase.txt"):
    text = extract_text_from_pdf(pdf_path)
    language = detect(text)
    print(f"Detected language: {language}")

    if language == 'ar':
        print("Detected Arabic. Running Arabic paraphrasing pipeline...")
        return paraphrase_arabic(pdf_path, text_output_path=text_output_path_ar)
    else:
        print("Detected English. Running English paraphrasing pipeline...")
        return paraphrase_english(text, text_output_path=text_output_path_en)

# API endpoint to upload PDF and paraphrase content
@app.route('/paraphrase', methods=['POST'])
def paraphrase_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        file.save(temp_pdf.name)
        pdf_path = temp_pdf.name

    try:
        # Run the summarization pipeline
        final_summary = detect_language_and_paraphrase(pdf_path)
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({'error': 'Summarization failed'}), 500
    finally:
        # Clean up the temp file
        os.remove(pdf_path)

    # Return the summary
    return jsonify({'summary': final_summary})

if __name__ == '__main__':
    app.run(debug=True)
