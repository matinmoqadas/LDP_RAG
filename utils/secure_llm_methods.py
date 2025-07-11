import os
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
import spacy


api_key = os.getenv('AVALAI_API_KEY')
base_url = "https://api.avalai.ir/v1"

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Function to initialize LLM with specified model name
def init_llm(model_name):
    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        api_key=api_key
    )

# Paraphrase method
def paraphrase(context, model_name="gemini-2.5-flash-preview-05-20"):
    llm = init_llm(model_name)
    prompt = f"""
    Given the following context, extract the useful or important part of the context.
    Remember, *DO NOT* edit the extracted parts of the context.

    > Context:
    >>>
    {context}
    >>>

    Extracted relevant parts:"""

    response = llm.invoke(prompt)
    return response.content.strip()

# ZeroGen method
def zerogen(context, model_name="gemini-2.5-flash-preview-05-20"):
    llm = init_llm(model_name)
    doc = nlp(context)
    entities = ', '.join(set([ent.text for ent in doc.ents]))

    prompt = f"""
    The context is: {context}.

    {entities} is the answer of the following question:"""

    response = llm.invoke(prompt)
    return response.content.strip()

# AttrPrompt method
def attrprompt(topic, model_name="gemini-2.5-flash-preview-05-20"):
    llm = init_llm(model_name)

    prompt_attrs = f"What do you think are the important attributes for generating {topic} data?"
    attributes = llm.invoke(prompt_attrs).content.strip().split('\n')[:5]

    subtopics = {}
    for attr in attributes:
        prompt_subtopics = f"Generate several different subtopics for the attribute: {attr}."
        subtopics[attr] = llm.invoke(prompt_subtopics).content.strip().split('\n')[:3]

    prompt_gen = "Generate a piece of content covering the following subtopics: " + ", ".join([st for sublist in subtopics.values() for st in sublist])
    content = llm.invoke(prompt_gen).content.strip()

    return content

# SAGE method
def sage(context, model_name="gemini-2.5-flash-preview-05-20"):
    llm = init_llm(model_name)

    prompt_phase1 = f"""
    Please summarize the key points from the following conversation:
    {context}

    Patient:
    [Clear Symptom Description]:
    [Medical History]:
    [Current Concerns]:
    [Recent Events]:
    [Specific Questions]:

    Doctor:
    [Clear Diagnosis or Assessment]:
    [Reassurance and Empathy]:
    [Treatment Options and Explanations]:
    [Follow-up and Next Steps]:
    [Education and Prevention]:"""

    summary = llm.invoke(prompt_phase1).content.strip()

    prompt_phase2 = f"""
    Here is a summary of the key points:
    {summary}

    Generate a SINGLE-ROUND patient–doctor medical dialog using ALL the key points provided.
    Include ONLY ONE question from the patient and ONE response from the doctor.

    Format:
    Patient: [Question contains ALL Patient key points provided]
    Doctor: [Response contains ALL Doctor key points provided]

    Do not generate any additional rounds of dialog."""

    dialog = llm.invoke(prompt_phase2).content.strip()

    return dialog

# Redact method
def redact(context):
    doc = nlp(context)
    for ent in doc.ents:
        context = context.replace(ent.text, 'IIIIII')
    return context

# TypedHolder method
def typedholder(context):
    doc = nlp(context)
    for ent in doc.ents:
        context = context.replace(ent.text, ent.label_)
    return context
