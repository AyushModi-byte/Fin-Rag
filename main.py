"""FinRag API entry point: `python main.py` (or `uvicorn main:app`)."""
import uvicorn

from src.api.routes import create_app
from src.config import load_config
from src.ingestion.loader import setup_logging

cfg = load_config()
setup_logging(cfg.data.log_file)
app = create_app(cfg=cfg)

if __name__ == "__main__":
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)
