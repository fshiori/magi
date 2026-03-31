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


# 50 curated questions across 5 categories (10 each)
# Including "Hell Difficulty" questions that often trip up LLMs.
BUILTIN_DATASET: list[BenchQuestion] = [
    # === STEM (10) ===
    BenchQuestion("What is the derivative of x^3?", ["x^2", "3x^2", "3x", "x^3/3"], 1, "stem"),
    BenchQuestion("Which planet is closest to the Sun?", ["Venus", "Mercury", "Mars", "Earth"], 1, "stem"),
    BenchQuestion("What is the chemical formula for water?", ["CO2", "H2O", "NaCl", "O2"], 1, "stem"),
    BenchQuestion("What is the speed of light in vacuum (approximately)?", ["300,000 m/s", "300,000 km/s", "3,000 km/s", "30,000 km/s"], 1, "stem"),
    BenchQuestion("In binary, what is 1010 + 0110?", ["1100", "10000", "1110", "10010"], 1, "stem"),
    # Harder STEM
    BenchQuestion(
        "Which of the following describes the 'Gibbs Free Energy' (G)?",
        ["G = H + TS", "G = H - TS", "G = U + PV", "G = U - TS"],
        1, "stem"
    ),
    BenchQuestion(
        "In quantum mechanics, what does the 'Heisenberg Uncertainty Principle' relate?",
        ["Mass and Energy", "Position and Momentum", "Time and Space", "Wave and Particle"],
        1, "stem"
    ),
    BenchQuestion(
        "What is the limit of (1 + 1/n)^n as n approaches infinity?",
        ["1", "Infinity", "e", "0"],
        2, "stem"
    ),
    BenchQuestion(
        "Which particle is the force carrier for the strong nuclear force in the Standard Model?",
        ["Photon", "W boson", "Gluon", "Z boson"],
        2, "stem"
    ),
    BenchQuestion(
        "What is the time complexity of finding the shortest path in a graph using Dijkstra's algorithm with a binary heap?",
        ["O(V^2)", "O(E log V)", "O(V log E)", "O(E + V)"],
        1, "stem"
    ),

    # === Humanities (10) ===
    BenchQuestion("Who wrote 'Romeo and Juliet'?", ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"], 1, "humanities"),
    BenchQuestion("In which year did World War II end?", ["1943", "1944", "1945", "1946"], 2, "humanities"),
    BenchQuestion("Which ancient civilization built the pyramids of Giza?", ["Romans", "Greeks", "Egyptians", "Persians"], 2, "humanities"),
    BenchQuestion("What is the official language of Brazil?", ["Spanish", "Portuguese", "French", "English"], 1, "humanities"),
    BenchQuestion("Who painted the Mona Lisa?", ["Michelangelo", "Raphael", "Leonardo da Vinci", "Donatello"], 2, "humanities"),
    # Harder Humanities
    BenchQuestion(
        "Which philosopher wrote 'The Critique of Pure Reason'?",
        ["David Hume", "Immanuel Kant", "Friedrich Nietzsche", "Jean-Paul Sartre"],
        1, "humanities"
    ),
    BenchQuestion(
        "In which century did the 'Great Schism' between the Eastern Orthodox and Roman Catholic churches occur?",
        ["9th", "10th", "11th", "12th"],
        2, "humanities"
    ),
    BenchQuestion(
        "Who was the primary author of the 'Declaration of Independence'?",
        ["George Washington", "Benjamin Franklin", "Thomas Jefferson", "John Adams"],
        2, "humanities"
    ),
    BenchQuestion(
        "Which art movement is Salvador Dalí most closely associated with?",
        ["Impressionism", "Cubism", "Surrealism", "Expressionism"],
        2, "humanities"
    ),
    BenchQuestion(
        "What was the capital of the Byzantine Empire?",
        ["Rome", "Athens", "Constantinople", "Alexandria"],
        2, "humanities"
    ),

    # === Social Sciences (10) ===
    BenchQuestion("What does GDP stand for?", ["Gross Domestic Product", "General Domestic Price", "Gross Development Plan", "General Development Product"], 0, "social_sciences"),
    BenchQuestion("Which economic system is characterized by private ownership and free markets?", ["Socialism", "Communism", "Capitalism", "Feudalism"], 2, "social_sciences"),
    BenchQuestion("What is the term for a prolonged period of economic decline?", ["Inflation", "Recession", "Deflation", "Stagnation"], 1, "social_sciences"),
    BenchQuestion("In psychology, what is the term for learning by observing others?", ["Classical conditioning", "Operant conditioning", "Social learning", "Habituation"], 2, "social_sciences"),
    BenchQuestion("What type of government is ruled by a single person with absolute power?", ["Democracy", "Oligarchy", "Autocracy", "Theocracy"], 2, "social_sciences"),
    # Harder Social Sciences
    BenchQuestion(
        "Who is considered the father of 'Modern Macroeconomics'?",
        ["Adam Smith", "Karl Marx", "John Maynard Keynes", "Milton Friedman"],
        2, "social_sciences"
    ),
    BenchQuestion(
        "In sociology, what does 'Anomie' refer to?",
        ["A state of social stability", "A breakdown of social norms", "The process of socialization", "Economic inequality"],
        1, "social_sciences"
    ),
    BenchQuestion(
        "Which psychological experiment demonstrated the power of authority/obedience?",
        ["Stanford Prison Experiment", "Milgram Experiment", "Asch Conformity Test", "Marshmallow Test"],
        1, "social_sciences"
    ),
    BenchQuestion(
        "What is the 'Prisoner's Dilemma' a standard example of?",
        ["Market failure", "Game Theory", "Behavioral economics", "Fiscal policy"],
        1, "social_sciences"
    ),
    BenchQuestion(
        "Which international organization was the predecessor to the United Nations?",
        ["League of Nations", "Warsaw Pact", "NATO", "European Union"],
        0, "social_sciences"
    ),

    # === Logic & Reasoning (10) ===
    BenchQuestion("If all roses are flowers and some flowers fade quickly, which must be true?", ["All roses fade quickly", "Some roses fade quickly", "No roses fade quickly", "None of these must be true"], 3, "logic"),
    BenchQuestion("What comes next in the sequence: 2, 6, 18, 54, ...?", ["108", "162", "72", "216"], 1, "logic"),
    BenchQuestion("If A > B and B > C, what is the relationship between A and C?", ["A < C", "A = C", "A > C", "Cannot determine"], 2, "logic"),
    BenchQuestion("A farmer has 17 sheep. All but 9 die. How many sheep are alive?", ["8", "9", "17", "0"], 1, "logic"),
    BenchQuestion("Which number does not belong: 2, 3, 5, 7, 11, 14?", ["2", "3", "14", "11"], 2, "logic"),
    # Harder Logic (The "Hell" part)
    BenchQuestion(
        "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",
        ["1", "2", "3", "4"],
        0, "logic"
    ),
    BenchQuestion(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        ["$0.10", "$0.05", "$0.01", "$0.11"],
        1, "logic"
    ),
    BenchQuestion(
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        ["100 minutes", "5 minutes", "20 minutes", "50 minutes"],
        1, "logic"
    ),
    BenchQuestion(
        "All bloops are razzies. All razzies are lala. Therefore, all bloops are lala. Is this argument valid?",
        ["Yes", "No", "Only if lala exists", "Cannot determine"],
        0, "logic"
    ),
    BenchQuestion(
        "If you flip a fair coin 3 times and it comes up heads all 3 times, what is the probability it will be heads on the 4th flip?",
        ["1/16", "1/2", "1/8", "1/4"],
        1, "logic"
    ),

    # === Clinical Knowledge (10) ===
    BenchQuestion("What organ produces insulin?", ["Liver", "Kidney", "Pancreas", "Spleen"], 2, "clinical"),
    BenchQuestion("What is the normal resting heart rate for adults?", ["40-60 bpm", "60-100 bpm", "100-120 bpm", "120-140 bpm"], 1, "clinical"),
    BenchQuestion("Which vitamin is primarily obtained from sunlight exposure?", ["Vitamin A", "Vitamin C", "Vitamin D", "Vitamin E"], 2, "clinical"),
    BenchQuestion("What is the largest organ in the human body?", ["Liver", "Brain", "Skin", "Heart"], 2, "clinical"),
    BenchQuestion("What type of blood cell fights infection?", ["Red blood cells", "White blood cells", "Platelets", "Plasma"], 1, "clinical"),
    # Harder Clinical
    BenchQuestion(
        "Which of the following is the most common cause of community-acquired pneumonia?",
        ["Staphylococcus aureus", "Streptococcus pneumoniae", "Mycoplasma pneumoniae", "Haemophilus influenzae"],
        1, "clinical"
    ),
    BenchQuestion(
        "In the ECG, what does the P wave represent?",
        ["Ventricular depolarization", "Atrial depolarization", "Ventricular repolarization", "Atrial repolarization"],
        1, "clinical"
    ),
    BenchQuestion(
        "What is the first-line treatment for an acute anaphylactic reaction?",
        ["Diphenhydramine", "Epinephrine", "Methylprednisolone", "Albuterol"],
        1, "clinical"
    ),
    BenchQuestion(
        "Which cranial nerve is responsible for the sense of smell?",
        ["CN I (Olfactory)", "CN II (Optic)", "CN III (Oculomotor)", "CN IV (Trochlear)"],
        0, "clinical"
    ),
    BenchQuestion(
        "What is the primary pathophysiology in Type 1 Diabetes Mellitus?",
        ["Insulin resistance", "Alpha cell overproduction", "Autoimmune destruction of beta cells", "Liver failure"],
        2, "clinical"
    ),
]


def get_dataset(name: str = "builtin") -> list[BenchQuestion]:
    """Get a benchmark dataset by name.

    Supports:
    - 'builtin': The hard-coded 50 questions.
    - 'mmlu:<category>': MMLU dataset from Hugging Face (e.g., mmlu:abstract_algebra).
    """
    if name == "builtin":
        return BUILTIN_DATASET

    if name.startswith("mmlu:"):
        category = name.split(":")[1]
        return load_mmlu(category)

    raise ValueError(f"Unknown dataset: {name}. Available: builtin, mmlu:<category>")


def load_mmlu(category: str, split: str = "test") -> list[BenchQuestion]:
    """Load MMLU dataset from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Dataset dependencies not installed. Run: pip install datasets pandas")

    print(f"Loading MMLU dataset category: {category}...")
    try:
        ds = load_dataset("cais/mmlu", category, split=split, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"Failed to load MMLU category '{category}': {e}")

    questions = []
    for item in ds:
        questions.append(BenchQuestion(
            question=item["question"],
            choices=item["choices"],
            correct=item["answer"],
            category=f"mmlu:{category}",
        ))

    return questions


def get_categories() -> list[str]:
    """Get all unique categories in the builtin dataset."""
    return sorted(set(q.category for q in BUILTIN_DATASET))
