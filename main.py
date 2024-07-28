from reval.reval import Reval
from reval.language_models import LanguageModel
from reval.functions import binary_judge, criteria_generator

if __name__ == "__main__":
    result = Reval(
        "tasks.csv", [LanguageModel("gpt-4o-mini")], binary_judge, criteria_generator
    )
