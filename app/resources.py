"""Trusted reading list shown under the assistant's answers.

Links are chosen by the app from the user's own answers, never by the language model,
so the assistant can't invent or alter a source. Every URL was checked on 22 Oct 2026.
Health services now call the condition PMOS (polyendocrine metabolic ovarian syndrome),
formerly PCOS, so several pages use the new name.
"""

from __future__ import annotations

RESOURCES = {
    "who": {"title": "Polycystic ovary syndrome: fact sheet", "source": "World Health Organization",
            "url": "https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome",
            "why": "What the condition is, how common it is, and why up to 70% of women are undiagnosed."},
    "hse_symptoms": {"title": "PMOS (formerly PCOS): symptoms", "source": "HSE Ireland",
                     "url": "https://www2.hse.ie/conditions/polycystic-ovary-syndrome/",
                     "why": "Common symptoms, including irregular periods, hair growth and weight changes."},
    "hse_diagnosis": {"title": "PMOS (formerly PCOS): diagnosis", "source": "HSE Ireland",
                      "url": "https://www2.hse.ie/conditions/polycystic-ovary-syndrome/diagnosis/",
                      "why": "How a GP diagnoses it, and the tests you may be offered."},
    "nhs": {"title": "PMOS (formerly PCOS)", "source": "NHS",
            "url": "https://www.nhs.uk/conditions/polycystic-ovary-syndrome-pcos/",
            "why": "Overview of symptoms, causes, treatment and when to see a GP."},
    "skin": {"title": "Acanthosis nigricans (dark skin patches)", "source": "NHS",
             "url": "https://www.nhs.uk/conditions/acanthosis-nigricans/",
             "why": "Dark, thick skin on the neck or armpits, and its link with diabetes and hormone conditions."},
    "hair": {"title": "Excessive hair growth (hirsutism)", "source": "NHS",
             "url": "https://www.nhs.uk/conditions/hirsutism/",
             "why": "Why extra face or body hair happens, and the treatments available."},
    "jean_hailes": {"title": "PMOS (formerly PCOS): symptoms, diagnosis and treatment", "source": "Jean Hailes for Women's Health",
                    "url": "https://www.jeanhailes.org.au/health-a-z/pcos",
                    "why": "Plain-language guide written with clinicians, including fertility and lifestyle."},
    "guideline": {"title": "Recommendations from the 2023 International Evidence-based Guideline for PCOS",
                  "source": "Teede et al., J Clin Endocrinol Metab, 2023",
                  "url": "https://doi.org/10.1210/clinem/dgad463",
                  "why": "The research guideline doctors follow (technical)."},
}

# Which resources answer which topic
TOPIC_RESOURCES = {
    "cycle": ["hse_symptoms", "nhs"], "skin": ["skin"], "hair": ["hair"],
    "weight": ["jean_hailes", "nhs"], "diagnosis": ["hse_diagnosis", "guideline"],
    "general": ["who", "hse_symptoms"],
}

QUESTION_TOPICS = {
    "cycle": ("cycle", "period", "irregular", "length"), "skin": ("skin", "dark", "neck", "armpit"),
    "hair": ("hair",), "weight": ("weight", "bmi", "diet", "exercise", "food"),
    "diagnosis": ("diagnos", "test", "doctor", "gp", "have pcos", "pmos", "do i have"),
}


def for_reading(reading: dict) -> list[dict]:
    """Up to 4 links matched to her own answers, most relevant first."""
    keys = []
    f, flags = reading["features"], reading["flags"]
    if flags["axis1_pattern"] or flags["regularity"] == "irregular":
        keys += TOPIC_RESOURCES["cycle"]
    if f.get("Skin darkening"):
        keys += TOPIC_RESOURCES["skin"]
    if f.get("hair growth"):
        keys += TOPIC_RESOURCES["hair"]
    if reading["level"] in ("L2", "S"):
        keys += TOPIC_RESOURCES["diagnosis"]
    keys += TOPIC_RESOURCES["general"]
    return [RESOURCES[k] for k in dict.fromkeys(keys)][:4]


def for_question(question: str) -> list[dict]:
    q = question.lower()
    keys = [k for topic, words in QUESTION_TOPICS.items() if any(w in q for w in words)
            for k in TOPIC_RESOURCES[topic]]
    return [RESOURCES[k] for k in dict.fromkeys(keys)][:3]


def as_markdown(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["**Read more (trusted sources)**"]
    lines += [f"- [{r['title']}]({r['url']}): {r['source']}. {r['why']}" for r in items]
    return "\n".join(lines)
