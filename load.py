from langchain_huggingface import HuggingFaceEndpoint

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = "your_huggingface_api_token_here"  # 🔴 Hardcoded token (not recommended for security)

llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"max_length": 512}
)
