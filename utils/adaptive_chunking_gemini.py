"GEMINI FRIENDLY"

import google.generativeai as genai
from docx import Document
import re
import json

# Define constants to avoid "magic strings"
CLASSIFICATION_MODEL_NAME = 'gemini-2.5-flash'
DEFAULT_TAG = 'Public'

class DocumentProcessor:
    """
    A class to process DOCX files by chunking, classifying,
    and saving the content.
    """
    def __init__(self, api_key):
        """Initializes the processor with the Generative AI model."""

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(CLASSIFICATION_MODEL_NAME)

    def _extract_text_from_docx(self, docx_path: str) -> str:
        """Extracts all text from a DOCX file."""
        try:
            document = Document(docx_path)
            return "\n".join(p.text for p in document.paragraphs)
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            return ""

    def _count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a given text."""
        try:
            return self.model.count_tokens(text).total_tokens
        except Exception as e:
            print(f"Error counting tokens: {e}. Using fallback estimate.")
            return len(text) // 4  # Rough estimate

    def _split_into_sentences(self, text: str) -> list[str]:
        """Splits text into sentences, ensuring punctuation."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        processed = []
        for sentence in sentences:
            stripped_sentence = sentence.strip()
            if stripped_sentence:
                if not re.search(r'[.!?]$', stripped_sentence):
                    stripped_sentence += '.'
                processed.append(stripped_sentence)
        return processed

    def _build_chunks(self, sentences: list[str], max_tokens: int) -> list[dict]:
        """Builds text chunks from sentences, respecting the token limit."""
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        sentence_ids = []
        labels = []

        for idx, sentence in enumerate(sentences):
        
            sentence_token_count = self._count_tokens(sentence)
            sentence_label = self._classify_sentence(sentence = sentence)
            print(f"----------sentence_label:{sentence_label}----------")
            labels.append(sentence_label)
            
            if current_token_count + sentence_token_count > max_tokens and current_chunk_sentences:
                chunks.append({
                    'chunk_id': len(chunks) + 1,
                    'tag': DEFAULT_TAG,
                    'content': ' '.join(current_chunk_sentences).strip(),
                    'source_sentence_ids': sentence_ids,
                    'labels' : labels
                })
                current_chunk_sentences, sentence_ids, current_token_count = [], [], 0

            current_chunk_sentences.append(sentence)
            sentence_ids.append(idx + 1)
            current_token_count += sentence_token_count
            # 
        print(f"len_idx:{len(sentence_ids)}, len_labels:{len(labels)}")
            # 
        if current_chunk_sentences:
            chunks.append({
                'chunk_id': len(chunks) + 1,
                'tag': DEFAULT_TAG,
                'content': ' '.join(current_chunk_sentences).strip(),
                'source_sentence_ids': sentence_ids,
                'labels' : labels
            })

        return chunks

    def _classify_sentence(self, sentence : str) -> str:
        """Classifies a sentence as Public, Sensitive, or Confidential."""
        prompt = (
            "Classify the text into: 'Public', 'Sensitive', or 'Confidential'. "
            "Respond with only the category name.\n\n"
            f"Text: \"\"\"{sentence}\"\"\""
        )
        break_flag = False
        tag = DEFAULT_TAG
        while (not break_flag):
          try:
              response = self.model.generate_content(prompt, generation_config={'temperature': 0.0})
              tag_tmp = response.text.strip()
              if tag_tmp in ['Public', 'Sensitive', 'Confidential']:
                tag = tag_tmp
                break_flag = True   
          except Exception as e:
              print(f'error {e} occured')

        return tag


    def _classify_chunk(self, chunk_content: str) -> str:
        """Classifies a chunk as Public, Sensitive, or Confidential."""
        prompt = (
            "Classify the text into: 'Public', 'Sensitive', or 'Confidential'. "
            "Respond with only the category name.\n\n"
            f"Text: \"\"\"{chunk_content}\"\"\""
        )
        try:
            response = self.model.generate_content(prompt, generation_config={'temperature': 0.0})
            tag = response.text.strip()
            return tag if tag in ['Public', 'Sensitive', 'Confidential'] else DEFAULT_TAG
        except Exception as e:
            print(f"Error classifying chunk: {e}. Defaulting to '{DEFAULT_TAG}'.")
            return DEFAULT_TAG

    def _save_to_json(self, chunks_data: list[dict], output_filename: str):
        """Saves the chunked data to a JSON file."""
        try:
            with open(output_filename, 'w', encoding='utf-8') as file:
                json.dump(chunks_data, file, indent=4, ensure_ascii=False)
            print(f"Chunked data saved to {output_filename}")
        except IOError as e:
            print(f"Error saving to JSON file: {e}")

    def process(self, docx_path: str, output_filename: str, max_tokens: int = 500):
        """The main method to process a DOCX file."""
        text = self._extract_text_from_docx(docx_path)
        if not text:
            print("No text to process. Exiting.")
            return

        sentences = self._split_into_sentences(text)
        chunks = self._build_chunks(sentences, max_tokens)

        for chunk in chunks:
            print(f"Classifying chunk {chunk['chunk_id']}...")
            tag = self._classify_chunk(chunk['content'])
            chunk['tag'] = tag
            # chunk['content'] = f'[{tag.upper()}]: {chunk["content"]}'
            chunk['labels'] = chunk['labels']

        self._save_to_json(chunks, output_filename)

