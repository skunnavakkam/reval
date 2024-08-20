import pandas as pd
from typing import List, Callable


def Reval(
    df: pd.DataFrame,
    models_to_judge: List[Callable],
    preprocessing: List[Callable],
    grader: Callable,
) -> pd.DataFrame:
    col = [c.lower() for c in df.columns]
    if "tasks" not in col:
        raise ValueError("Tasks column not found in the dataframe")

    # need to get "tasks", "success criteria", "failure criteria", "correct answer", "good example", "bad example"
    # these may not exist

    for _, row in df.iterrows():
        task_data = (
            row["tasks"],
            row.get("success criteria", None),
            row.get("failure criteria", None),
            row.get("correct answer", None),
            row.get("good example", None),
            row.get("bad example", None),
        )

        for step in preprocessing:
            task_data = step(*task_data)

        model_responses = []
        for model in models_to_judge:
            response = model(task_data[0])
            model_responses.append(response)

        grades = grader(*task_data, model_responses)
