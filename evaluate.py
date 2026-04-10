from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_chatbot.pipeline import evaluate_command


if __name__ == "__main__":
    evaluate_command()
