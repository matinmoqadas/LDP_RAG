from openai import OpenAI
from docx import Document
import re
import json
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple

CLASSIFICATION_MODEL_NAME = 'gpt-4o-mini'
DEFAULT_TAG = 'Public'

class SemanticDocumentProcessor:

    def __init__(self, api_key, base_url="https://api.avalai.ir/v1"):
        """Initializes the processor with the OpenAI API and semantic model."""
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # Load sentence transformer for semantic similarity
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

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
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"Error counting tokens: {e}. Using fallback estimate.")
            return len(text) // 4

    def _split_into_sentences(self, text: str) -> List[str]:
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

    def _get_semantic_similarity(self, sentences: List[str]) -> np.ndarray:
        """Compute semantic similarity matrix for sentences."""
        try:
            embeddings = self.semantic_model.encode(sentences)
            similarity_matrix = cosine_similarity(embeddings)
            return similarity_matrix
        except Exception as e:
            print(f"Error computing semantic similarity: {e}")
            return np.eye(len(sentences))  # Return identity matrix as fallback

    def _find_semantic_boundaries(self, sentences: List[str], similarity_threshold: float = 0.5) -> List[int]:
        """Find semantic boundaries between sentences based on similarity."""
        if len(sentences) <= 1:
            return []

        similarity_matrix = self._get_semantic_similarity(sentences)
        boundaries = []

        for i in range(len(sentences) - 1):
            # Check similarity between consecutive sentences
            similarity = similarity_matrix[i][i + 1]
            if similarity < similarity_threshold:
                boundaries.append(i + 1)  # Boundary after sentence i

        return boundaries

    def _build_semantic_chunks(self, sentences: List[str], max_tokens: int,
                             similarity_threshold: float = 0.5) -> List[Dict]:
        """Build chunks based on semantic boundaries while respecting token limits."""
        if not sentences:
            return []

        # Find potential semantic boundaries
        semantic_boundaries = set(self._find_semantic_boundaries(sentences, similarity_threshold))

        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        current_sentence_ids = []
        current_labels = []

        for idx, sentence in enumerate(sentences):
            sentence_token_count = self._count_tokens(sentence)
            sentence_label = self._classify_sentence(sentence)
            print(f"Processing sentence {idx + 1}: {sentence_label}")

            # Check if we need to create a new chunk
            should_break = False

            if current_chunk_sentences:
                # Check if adding this sentence would exceed token limit
                if current_token_count + sentence_token_count > max_tokens:
                    # If we're at a semantic boundary, break here
                    if idx in semantic_boundaries:
                        should_break = True
                    # If we're way over the limit, force a break
                    elif current_token_count + sentence_token_count > max_tokens * 1.2:
                        should_break = True
                    # Otherwise, try to find the nearest semantic boundary
                    else:
                        # Look ahead for a nearby semantic boundary
                        nearby_boundary = None
                        for boundary in semantic_boundaries:
                            if idx <= boundary <= min(idx + 3, len(sentences)):
                                nearby_boundary = boundary
                                break

                        if nearby_boundary and nearby_boundary == idx:
                            should_break = True

            if should_break and current_chunk_sentences:
                # Create chunk with current sentences
                sentence_labels = {}
                for i, (sent_id, label) in enumerate(zip(current_sentence_ids, current_labels)):
                    sentence_labels[f"sentence {i+1}"] = label.upper()

                chunks.append({
                    'chunk_id': len(chunks) + 1,
                    'tag': self._determine_highest_tag(current_labels),
                    'content': ' '.join(current_chunk_sentences).strip(),
                    'sentence_labels': sentence_labels,
                    'semantic_coherence_score': self._calculate_chunk_coherence(current_chunk_sentences)
                })

                # Reset for new chunk
                current_chunk_sentences = []
                current_sentence_ids = []
                current_labels = []
                current_token_count = 0

            # Add current sentence to chunk
            current_chunk_sentences.append(sentence)
            current_sentence_ids.append(idx + 1)
            current_labels.append(sentence_label)
            current_token_count += sentence_token_count

        # Handle the last chunk
        if current_chunk_sentences:
            sentence_labels = {}
            for i, (sent_id, label) in enumerate(zip(current_sentence_ids, current_labels)):
                sentence_labels[f"sentence {i+1}"] = label.upper()

            chunks.append({
                'chunk_id': len(chunks) + 1,
                'tag': self._determine_highest_tag(current_labels),
                'content': ' '.join(current_chunk_sentences).strip(),
                'sentence_labels': sentence_labels,
                'semantic_coherence_score': self._calculate_chunk_coherence(current_chunk_sentences)
            })

        return chunks

    def _calculate_chunk_coherence(self, sentences: List[str]) -> float:
        """Calculate the semantic coherence score for a chunk."""
        if len(sentences) <= 1:
            return 1.0

        try:
            similarity_matrix = self._get_semantic_similarity(sentences)
            # Calculate average pairwise similarity
            mask = np.triu(np.ones_like(similarity_matrix), k=1).astype(bool)
            avg_similarity = similarity_matrix[mask].mean()
            return float(avg_similarity)
        except Exception as e:
            print(f"Error calculating coherence: {e}")
            return 0.5

    def _determine_highest_tag(self, labels: List[str]) -> str:
        """Determines the highest security level tag from a list of labels."""
        if 'Confidential' in labels:
            return 'CONFIDENTIAL'
        elif 'Sensitive' in labels:
            return 'SENSITIVE'
        else:
            return 'PUBLIC'

    def _classify_sentence(self, sentence: str) -> str:
        """Classifies a sentence as Public, Sensitive, or Confidential."""
        prompt = """Classify the following text into one of three security levels:

        CLASSIFICATION CRITERIA:
        - Public: General information, publicly available knowledge, educational content, non-specific data
        - Sensitive: Internal information, specific procedures, mild personal details, internal policies
        - Confidential: Personal identifiable information (PII), financial data, medical records, proprietary information, specific patient cases with identifying details

        Respond with only one word: 'Public', 'Sensitive', or 'Confidential'.

        Text: \"\"\"{}\"\"\"""".format(sentence)

        try:
            response = self.client.chat.completions.create(
                model=CLASSIFICATION_MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0
            )
            tag = response.choices[0].message.content.strip()
            if tag in ['Public', 'Sensitive', 'Confidential']:
                return tag
            else:
                return DEFAULT_TAG
        except Exception as e:
            print(f"Error classifying sentence: {e}. Defaulting to '{DEFAULT_TAG}'.")
            return DEFAULT_TAG

    def _save_to_json(self, chunks_data: List[Dict], output_filename: str):
        """Saves the chunked data to a JSON file."""
        try:
            with open(output_filename, 'w', encoding='utf-8') as file:
                json.dump(chunks_data, file, indent=4, ensure_ascii=False)
            print(f"Chunked data saved to {output_filename}")
        except IOError as e:
            print(f"Error saving to JSON file: {e}")

    def process(self, docx_path: str, output_filename: str, max_tokens: int = 500,
                similarity_threshold: float = 0.5):
        """The main method to process a DOCX file with semantic chunking."""
        text = self._extract_text_from_docx(docx_path)
        if not text:
            print("No text to process. Exiting.")
            return

        sentences = self._split_into_sentences(text)
        chunks = self._build_semantic_chunks(sentences, max_tokens, similarity_threshold)

        self._save_to_json(chunks, output_filename)

        # Print summary statistics
        total_sentences = sum(len(chunk['sentence_labels']) for chunk in chunks)
        public_count = sum(1 for chunk in chunks for label in chunk['sentence_labels'].values() if label == 'PUBLIC')
        sensitive_count = sum(1 for chunk in chunks for label in chunk['sentence_labels'].values() if label == 'SENSITIVE')
        confidential_count = sum(1 for chunk in chunks for label in chunk['sentence_labels'].values() if label == 'CONFIDENTIAL')

        avg_coherence = np.mean([chunk['semantic_coherence_score'] for chunk in chunks])

        print(f"\n--- Classification Summary ---")
        print(f"Total sentences: {total_sentences}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Average semantic coherence: {avg_coherence:.3f}")
        print(f"PUBLIC: {public_count}")
        print(f"SENSITIVE: {sensitive_count}")
        print(f"CONFIDENTIAL: {confidential_count}")


# Alternative approach using topic modeling
class TopicBasedChunker(SemanticDocumentProcessor):
    """Enhanced version using topic modeling for better semantic boundaries."""

    def __init__(self, api_key, base_url="https://api.avalai.ir/v1"):
        super().__init__(api_key, base_url)

    def _identify_topics(self, sentences: List[str]) -> List[str]:
        """Identify topics for sentences using LLM."""
        if not sentences:
            return []

        # Process in batches to avoid token limits
        topics = []
        batch_size = 10

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_text = "\n".join([f"{idx+1}. {sent}" for idx, sent in enumerate(batch)])

            prompt = f"""Analyze the following sentences and assign a brief topic/theme to each (1-3 words max).

            Sentences:
            {batch_text}

            Respond in JSON format:
            {{"topics": ["topic1", "topic2", ...]}}"""

            try:
                response = self.client.chat.completions.create(
                    model=CLASSIFICATION_MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1
                )

                result = json.loads(response.choices[0].message.content.strip())
                topics.extend(result.get("topics", ["general"] * len(batch)))

            except Exception as e:
                print(f"Error identifying topics: {e}")
                topics.extend(["general"] * len(batch))

        return topics

    def _find_topic_boundaries(self, topics: List[str]) -> List[int]:
        """Find boundaries where topics change."""
        boundaries = []
        for i in range(len(topics) - 1):
            if topics[i] != topics[i + 1]:
                boundaries.append(i + 1)
        return boundaries


