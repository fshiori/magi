"""Built-in benchmark datasets.

MVP: A small curated set of questions with known answers.
Future: Download MMLU subsets from Hugging Face.
"""
from dataclasses import dataclass


@dataclass
class BenchQuestion:
    question: str
    choices: list[str]  # A, B, C, D
    correct: int  # index into choices (0-3)
    category: str


# 25 curated questions across 5 categories (5 each)
# These are MMLU-style multiple choice questions
BUILTIN_DATASET: list[BenchQuestion] = [
    # === STEM (5) ===
    BenchQuestion(
        "What is the derivative of x^3?",
        ["x^2", "3x^2", "3x", "x^3/3"],
        1, "stem",
    ),
    BenchQuestion(
        "Which planet is closest to the Sun?",
        ["Venus", "Mercury", "Mars", "Earth"],
        1, "stem",
    ),
    BenchQuestion(
        "What is the chemical formula for water?",
        ["CO2", "H2O", "NaCl", "O2"],
        1, "stem",
    ),
    BenchQuestion(
        "What is the speed of light in vacuum (approximately)?",
        ["300,000 m/s", "300,000 km/s", "3,000 km/s", "30,000 km/s"],
        1, "stem",
    ),
    BenchQuestion(
        "In binary, what is 1010 + 0110?",
        ["1100", "10000", "1110", "10010"],
        1, "stem",
    ),
    # === Humanities (5) ===
    BenchQuestion(
        "Who wrote 'Romeo and Juliet'?",
        ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
        1, "humanities",
    ),
    BenchQuestion(
        "In which year did World War II end?",
        ["1943", "1944", "1945", "1946"],
        2, "humanities",
    ),
    BenchQuestion(
        "Which ancient civilization built the pyramids of Giza?",
        ["Romans", "Greeks", "Egyptians", "Persians"],
        2, "humanities",
    ),
    BenchQuestion(
        "What is the official language of Brazil?",
        ["Spanish", "Portuguese", "French", "English"],
        1, "humanities",
    ),
    BenchQuestion(
        "Who painted the Mona Lisa?",
        ["Michelangelo", "Raphael", "Leonardo da Vinci", "Donatello"],
        2, "humanities",
    ),
    # === Social Sciences (5) ===
    BenchQuestion(
        "What does GDP stand for?",
        ["Gross Domestic Product", "General Domestic Price", "Gross Development Plan", "General Development Product"],
        0, "social_sciences",
    ),
    BenchQuestion(
        "Which economic system is characterized by private ownership and free markets?",
        ["Socialism", "Communism", "Capitalism", "Feudalism"],
        2, "social_sciences",
    ),
    BenchQuestion(
        "What is the term for a prolonged period of economic decline?",
        ["Inflation", "Recession", "Deflation", "Stagnation"],
        1, "social_sciences",
    ),
    BenchQuestion(
        "In psychology, what is the term for learning by observing others?",
        ["Classical conditioning", "Operant conditioning", "Social learning", "Habituation"],
        2, "social_sciences",
    ),
    BenchQuestion(
        "What type of government is ruled by a single person with absolute power?",
        ["Democracy", "Oligarchy", "Autocracy", "Theocracy"],
        2, "social_sciences",
    ),
    # === Logic & Reasoning (5) ===
    BenchQuestion(
        "If all roses are flowers and some flowers fade quickly, which must be true?",
        ["All roses fade quickly", "Some roses fade quickly", "No roses fade quickly", "None of these must be true"],
        3, "logic",
    ),
    BenchQuestion(
        "What comes next in the sequence: 2, 6, 18, 54, ...?",
        ["108", "162", "72", "216"],
        1, "logic",
    ),
    BenchQuestion(
        "If A > B and B > C, what is the relationship between A and C?",
        ["A < C", "A = C", "A > C", "Cannot determine"],
        2, "logic",
    ),
    BenchQuestion(
        "A farmer has 17 sheep. All but 9 die. How many sheep are alive?",
        ["8", "9", "17", "0"],
        1, "logic",
    ),
    BenchQuestion(
        "Which number does not belong: 2, 3, 5, 7, 11, 14?",
        ["2", "3", "14", "11"],
        2, "logic",
    ),
    # === Clinical Knowledge (5) ===
    BenchQuestion(
        "What organ produces insulin?",
        ["Liver", "Kidney", "Pancreas", "Spleen"],
        2, "clinical",
    ),
    BenchQuestion(
        "What is the normal resting heart rate for adults?",
        ["40-60 bpm", "60-100 bpm", "100-120 bpm", "120-140 bpm"],
        1, "clinical",
    ),
    BenchQuestion(
        "Which vitamin is primarily obtained from sunlight exposure?",
        ["Vitamin A", "Vitamin C", "Vitamin D", "Vitamin E"],
        2, "clinical",
    ),
    BenchQuestion(
        "What is the largest organ in the human body?",
        ["Liver", "Brain", "Skin", "Heart"],
        2, "clinical",
    ),
    BenchQuestion(
        "What type of blood cell fights infection?",
        ["Red blood cells", "White blood cells", "Platelets", "Plasma"],
        1, "clinical",
    ),
]


def get_dataset(name: str = "builtin") -> list[BenchQuestion]:
    """Get a benchmark dataset by name."""
    if name == "builtin":
        return BUILTIN_DATASET
    raise ValueError(f"Unknown dataset: {name}. Available: builtin")


def get_categories() -> list[str]:
    """Get all unique categories in the builtin dataset."""
    return sorted(set(q.category for q in BUILTIN_DATASET))
