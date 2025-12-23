import json
from .llm_client import call_llm
from .utils import load_prompt

def analyze_text(text: str) -> dict:
    prompt_template = load_prompt(
        section="text_moderation",
        key="analysis_prompt"
    )

    prompt = prompt_template.replace("{{ text }}", text)
    raw_response = call_llm(prompt)

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        raise ValueError("Réponse LLM non valide")

    # Score de risque simple
    score = 0.2
    if data["viralite"] == "moyenne":
        score += 0.3
    elif data["viralite"] == "forte":
        score += 0.5

    if data["discours_haineux"]:
        score += 0.4

    data["risque_score"] = min(score, 1.0)
    return data
