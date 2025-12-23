import yaml
from pathlib import Path

PROMPT_PATH = Path(__file__).parent / "template.yml"

def load_prompt(section: str, key: str) -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        return data[section][key]
    except KeyError:
        raise ValueError(f"Prompt '{section}.{key}' introuvable")
