"""Lazy provider construction; custom endpoints are always explicit."""
import os


def make_embeddings(provider, *, api_key=None, base_url=None, model=None, device=None):
    try:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            options = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
            if base_url:
                options["base_url"] = base_url
            if model:
                options["model"] = model
            return OpenAIEmbeddings(**options)
        if provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            if not model:
                raise ValueError("embedding_model_name is required for Gemini embeddings")
            return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        if provider == "local":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model or "all-MiniLM-L6-v2", model_kwargs={"device": device or "cpu"})
    except ImportError as exc:
        raise ImportError(f"Install ldp-rag[{provider}] to use this embedding provider") from exc
    raise ValueError("embedding_model must be 'openai', 'gemini', or 'local'")


def make_llm(provider, *, api_key=None, base_url=None, model=None, temperature=0):
    if provider == "none":
        return None
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            options = {"api_key": api_key or os.getenv("OPENAI_API_KEY"), "temperature": temperature}
            if base_url:
                options["base_url"] = base_url
            return ChatOpenAI(model=model or "gpt-4o-mini", **options)
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            if not model:
                raise ValueError("model_name is required for Gemini")
            return ChatGoogleGenerativeAI(model=model, google_api_key=api_key or os.getenv("GOOGLE_API_KEY"), temperature=temperature)
    except ImportError as exc:
        raise ImportError(f"Install ldp-rag[{provider}] to use this chat provider") from exc
    raise ValueError("llm_name must be 'openai', 'gemini', or 'none'")


def invoke_text(llm, prompt, system_prompt=None):
    if llm is None:
        return f"[LLM stub]\n{prompt}"
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    content = llm.invoke(messages).content
    if not isinstance(content, str):
        raise TypeError("Expected a text response from the chat model")
    return content.strip()
