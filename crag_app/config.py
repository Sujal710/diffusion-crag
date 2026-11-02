"""Central settings, read from environment variables / .env file."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Groq ---
    groq_api_key: str
    # Big model writes the answer, small fast model does the grading/rewriting.
    groq_generator_model: str = "openai/gpt-oss-120b"
    groq_grader_model: str = "openai/gpt-oss-20b"

    # --- Pinecone (serverless) ---
    pinecone_api_key: str
    pinecone_index: str = "crag-docs"      # lowercase letters, numbers, hyphens only
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"            # free Starter plan: aws / us-east-1
    pinecone_region: str = "us-east-1"
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # runs locally via FastEmbed

    # --- Web search (used when retrieval is Incorrect / Ambiguous) ---
    tavily_api_key: str

    # --- Text -> video (open-source Stable Diffusion served by Hugging Face; nothing runs locally) ---
    image_provider: str = "cloudflare"   # "cloudflare" (free daily allowance) or "huggingface"

    # Cloudflare Workers AI: dash.cloudflare.com -> Workers AI -> Use REST API
    cf_account_id: str = ""
    cf_api_token: str = ""         # token with the "Workers AI" permission
    # SDXL Lightning = SDXL distilled to need ~4 steps (about 2 s per image). The standard
    # "@cf/stabilityai/stable-diffusion-xl-base-1.0" timed out on the free tier.
    cf_image_model: str = "@cf/bytedance/stable-diffusion-xl-lightning"
    cf_steps: int = 6              # Lightning: 4-8 is plenty (max 20)

    hf_token: str = ""             # huggingface.co/settings/tokens (needs "Inference Providers" permission)
    hf_image_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    hf_provider: str = "auto"      # which provider serves the model: auto / hf-inference / fal-ai / together ...
    video_width: int = 1024        # keep both multiples of 64
    video_height: int = 576

    # --- Login (JWT) ---
    jwt_secret: str = ""           # long random string; if empty a temporary one is used (logins reset on restart)
    jwt_expire_minutes: int = 1440 # how long a login stays valid (24 h)
    allow_signup: bool = True      # set false to stop new accounts being created

    # --- CRAG knobs ---
    top_k: int = 5
    upper_threshold: float = 0.7   # any doc >= this  -> CORRECT
    lower_threshold: float = 0.3   # all docs < this  -> INCORRECT, else AMBIGUOUS
    web_results: int = 5
    max_attempts: int = 3          # how many times the whole pipeline may run


@lru_cache
def get_settings() -> Settings:
    return Settings()
