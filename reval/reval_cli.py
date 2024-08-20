import pandas as pd
from reval import Reval, AsyncReval
from reval.language_models import GenericLanguageModel, AsyncGenericLanguageModel
from reval.graders import (
    CorrectAnswerGrader,
    AsyncCorrectAnswerGrader,
    CriteriaBinaryGrader,
    AsyncCriteriaBinaryGrader,
)
from reval.preprocessing import (
    CriteriaGeneratorWithExamples,
    AsyncCriteriaGeneratorWithExamples,
    CriteriaGenerator,
    AsyncCriteriaGenerator,
)

import argparse
from datetime import datetime


def reval_cli():
    parser = argparse.ArgumentParser(description="Reval")
    parser.add_argument(
        "path_to_tasks",
        type=str,
        help="Path to the tasks file. Reval looks for a column named 'tasks', and optionally columns named 'Success Criteria', 'Failure Criteria', 'Correct Answer', 'Good Example', and 'Bad Example'. You can have any of these, or none.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of models to evaluate",
    )
    parser.add_argument(
        "--rate_limit",
        type=float,
        default=None,
        help="Rate limit in seconds for API calls. If not provided, will not be run asynchronously.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the output CSV file.",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.path_to_tasks)
    cols = [c.lower().strip() for c in df.columns]
    if "tasks" not in cols:
        raise ValueError("Tasks column not found in the dataframe")

    disambiguations = {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    }
    if args.rate_limit is not None:
        preprocessing = []
        if "correct answer" in cols:
            grading_function = AsyncCorrectAnswerGrader()
        elif "success criteria" in cols and "failure criteria" in cols:
            grading_function = AsyncCriteriaBinaryGrader
        elif "good example" in cols or "bad example" in cols:
            preprocessing.append(AsyncCriteriaGeneratorWithExamples())
            grading_function = AsyncCriteriaBinaryGrader
        else:
            preprocessing.append(AsyncCriteriaGenerator())
            grading_function = AsyncCriteriaBinaryGrader

        models = [
            AsyncGenericLanguageModel(disambiguations.get(model, model))
            for model in args.models
        ]

        ret_df = AsyncReval(
            df, models, grading_function, preprocessing, args.rate_limit
        )

    else:
        preprocessing = []
        if "correct answer" in cols:
            grading_function = CorrectAnswerGrader()
        elif "success criteria" in cols and "failure criteria" in cols:
            grading_function = CriteriaBinaryGrader()
        elif "good example" in cols or "bad example" in cols:
            preprocessing.append(CriteriaGeneratorWithExamples())
            grading_function = CriteriaBinaryGrader()
        else:
            preprocessing.append(CriteriaGenerator())
            grading_function = CriteriaBinaryGrader()

        models = [
            GenericLanguageModel(disambiguations.get(model, model))
            for model in args.models
        ]

        ret_df = Reval(df, models, grading_function, preprocessing)

    if args.output_path is not None:
        ret_df.to_csv(args.output_path, index=False)
    else:
        ret_df.to_csv(
            f"reval_output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
            index=False,
        )
