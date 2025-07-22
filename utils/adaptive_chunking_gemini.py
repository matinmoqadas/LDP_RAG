import google.generativeai as genai
from docx import Document
import re
import json # Import the json module


genai.configure(api_key='')

# Initialize the Gemini model for classification and token counting
CLASSIFICATION_MODEL_NAME = 'gemini-2.5-flash'
classification_model = genai.GenerativeModel(CLASSIFICATION_MODEL_NAME)

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    """Extracts all text from a DOCX file."""
    try:
        document = Document(docx_path)
        text = []
        for paragraph in document.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

# Token counting function using Gemini's model
def count_tokens(text):
    """Returns the number of tokens in the input text using Gemini's model."""
    try:
        # Use the model's built-in token counting method
        response = classification_model.count_tokens(text)
        print(f"DEBUG: Token count for '{text[:50]}...' (API): {response.total_tokens}")
        return response.total_tokens
    except Exception as e:
        print(f"DEBUG: Error counting tokens with Gemini API: {e}. Using fallback estimate.")
        # Fallback to a simple character-based estimate if API call fails
        estimated_tokens = len(text) // 4 # Rough estimate: 1 token ~ 4 characters
        print(f"DEBUG: Token count for '{text[:50]}...' (Fallback): {estimated_tokens}")
        return estimated_tokens

# Function to chunk text while respecting token limit and sentence boundaries
def chunk_text_by_tokens(text, max_tokens=500):
    """Chunks the text into chunks of sentences while ensuring each chunk doesn't exceed the max token limit."""
    # A more robust sentence split using regex to handle various punctuation and spaces
    # Ensures that the delimiter (e.g., '.', '!', '?') is included at the end of the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Add back the delimiter if it was removed by split, or ensure it's there
    processed_sentences = []
    for s in sentences:
        s_stripped = s.strip()
        if s_stripped: # Ensure sentence is not empty
            # If the sentence doesn't end with a common punctuation, add a period for consistency
            if not re.search(r'[.!?]$', s_stripped):
                s_stripped += '.'
            processed_sentences.append(s_stripped)

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    sentence_ids = []

    for idx, sentence in enumerate(processed_sentences):
        # Count tokens for the current sentence
        sentence_token_count = count_tokens(sentence)

        print(f"DEBUG: Processing sentence {idx+1}: '{sentence[:50]}...' (Tokens: {sentence_token_count})")
        print(f"DEBUG: Current chunk tokens: {current_token_count}, Max tokens: {max_tokens}")

        # If adding the sentence would exceed the limit, finalize the current chunk
        # This condition ensures a new chunk is created if the current one is too large,
        # or if the very first sentence itself is larger than max_tokens.
        if current_token_count + sentence_token_count > max_tokens and current_chunk_sentences:
            # Finalize the current chunk
            chunks.append({
                'chunk_id': len(chunks) + 1,
                'tag': 'Public',  # Placeholder tag, will be updated later
                'content': ' '.join(current_chunk_sentences).strip(),
                'source_sentence_ids': sentence_ids
            })
            print(f"DEBUG: Chunk {len(chunks)} finalized. Starting new chunk.")
            # Start a new chunk with the current sentence
            current_chunk_sentences = [sentence]
            sentence_ids = [idx + 1]  # sentence IDs start from 1
            current_token_count = sentence_token_count
        elif sentence_token_count > max_tokens:
            # Handle case where a single sentence is larger than max_tokens
            # It will form its own chunk. This might lead to a chunk exceeding max_tokens,
            # but it respects sentence boundaries as requested.
            chunks.append({
                'chunk_id': len(chunks) + 1,
                'tag': 'Public',
                'content': sentence,
                'source_sentence_ids': [idx + 1]
            })
            print(f"DEBUG: Sentence {idx+1} (Tokens: {sentence_token_count}) > Max tokens. Creating single-sentence chunk.")
            current_chunk_sentences = [] # Reset for next sentence
            sentence_ids = []
            current_token_count = 0
        else:
            # Add the sentence to the current chunk
            current_chunk_sentences.append(sentence)
            sentence_ids.append(idx + 1)
            current_token_count += sentence_token_count

        print(f"DEBUG: After processing sentence {idx+1}: Current chunk tokens: {current_token_count}")

    # Add the last chunk if it contains content
    if current_chunk_sentences:
        chunks.append({
            'chunk_id': len(chunks) + 1,
            'tag': 'Public',  # Placeholder tag, will be updated later
            'content': ' '.join(current_chunk_sentences).strip(),
            'source_sentence_ids': sentence_ids
        })
        print(f"DEBUG: Final chunk {len(chunks)} added.")

    return chunks

# Function to classify each chunk using Gemini
def classify_chunk(chunk_content):
    """Classifies a chunk using the Gemini API."""
    prompt = (
        "Classify the following text into one of these categories: 'Public', 'Sensitive', or 'Confidential'. "
        "Provide only the category name as your response, without any additional text or explanation.\n\n"
        f"Text: \"\"\"{chunk_content}\"\"\""
    )

    try:
        response = classification_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=10, # Keep output short for classification
                temperature=0.0,      # Aim for deterministic classification
            )
        )
        # Extract the text from the response
        if response.candidates:
            # Access the text directly from the first part of the first candidate
            tag = response.candidates[0].content.parts[0].text.strip()
            # Basic validation to ensure the tag is one of the expected values
            if tag in ['Public', 'Sensitive', 'Confidential']:
                return tag
            else:
                print(f"Warning: Unexpected classification result: '{tag}'. Defaulting to 'Public'.")
                return 'Public'
        else:
            print("Warning: No candidates found in Gemini API response. Defaulting to 'Public'.")
            return 'Public'
    except Exception as e:
        print(f"Error classifying chunk with Gemini API: {e}. Defaulting to 'Public'.")
        return 'Public'

# Function to save chunked data to a JSON file
def save_chunks_to_json(chunks_data, output_filename="chunked_data.json"):
    """Saves the list of chunk dictionaries to a JSON file."""
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=4, ensure_ascii=False)
        print(f"Chunked data successfully saved to {output_filename}")
    except Exception as e:
        print(f"Error saving chunked data to JSON file: {e}")

# Main function to process DOCX, chunk the text, and tag the chunks
def process_document(docx_path, max_tokens=500):
    """Processes a DOCX file, chunking it and classifying each chunk."""
    # Step 1: Extract text from the DOCX file
    text = extract_text_from_docx(docx_path)
    if not text:
        print("No text extracted from the document. Exiting.")
        return []

    # Step 2: Chunk the text into manageable parts
    chunks = chunk_text_by_tokens(text, max_tokens)

    # Step 3: Classify each chunk and assign the appropriate tag
    chunk_metadata = []
    for chunk in chunks:
        print(f"Classifying chunk {chunk['chunk_id']}...")
        tag = classify_chunk(chunk['content'])
        chunk['tag'] = tag  # Update the chunk's tag
        chunk_metadata.append(chunk)

    return chunk_metadata
