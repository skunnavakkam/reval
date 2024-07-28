from typing import List, Callable, Dict, Any, Literal
import pandas as pd
from tqdm import tqdm
import random
import argparse
from .language_models import LanguageModel
import json
import numpy as np
from .functions import binary_judge, criteria_generator

# MARK: Criteria Generator


def Reval(
    task_location,  # csv file OR pandas dataframe
    models_to_eval,
    judging_model,
    criteria_model,
    mode="single",
    processed_tasks_path="processed_tasks.csv",
    results_path="results.csv",
) -> Dict[str, Any]:
    """
    Evaluate models on tasks specified in a CSV file.

    Args:
        task_path (str): Path to the CSV file containing tasks to evaluate.
        models (List[Callable]): List of model functions to be evaluated.
        mode (Literal["single", "arena"]): Evaluation mode, either "single" or "arena".
        processed_tasks_path (str): Path to save processed tasks.
        results_path (str): Path to save evaluation results.
        raw_data_path (str): Path to save raw evaluation data.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation results. The structure of this
                        dictionary may vary based on the evaluation mode and models used.

    This function reads tasks from a CSV file, processes them, evaluates the specified
    models on these tasks, and saves the results and raw data to the specified paths.
    The evaluation can be performed in either "single" mode (evaluating each model
    independently) or "arena" mode (comparing models against each other).
    """

    if isinstance(task_location, str):
        tasks = pd.read_csv(task_location)
    elif isinstance(task_location, pd.DataFrame):
        tasks = task_location

    columns = tasks.columns

    if "tasks" not in columns:
        raise KeyError(f"Column 'tasks' not found in {task_location}")

    # add columns success_criteria, failure_criteria, good_output, bad_output to the dataframe, and set all their values to None if they don't already exist
    if "success_criteria" not in columns:
        tasks["success_criteria"] = None
    if "failure_criteria" not in columns:
        tasks["failure_criteria"] = None

    for model in models_to_eval:
        if f"{model.name}.output" not in columns:
            tasks[f"{model.name}.output"] = None
        if f"{model.name}.grade" not in columns:
            tasks[f"{model.name}.grade"] = None

    for idx, row in tqdm(tasks.iterrows()):
        if row.get("tasks") is None:
            Warning(f"Skipping row {idx} as it has no tasks")
            continue

        task = row.get("tasks")
        success_criteria = row.get("success_criteria")
        failure_criteria = row.get("failure_criteria")
        good_output = row.get("good_output")
        bad_output = row.get("bad_output")

        success_criteria, failure_criteria = criteria_model(
            task, success_criteria, failure_criteria, good_output, bad_output
        )

        row["success_criteria"] = success_criteria
        row["failure_criteria"] = failure_criteria

        if mode == "single":
            for models in models_to_eval:
                model_output = model.get_generation(task)

                row[f"{model.name}.output"] = model_output

                grading_result = judging_model(
                    tasks, success_criteria, failure_criteria, model_output
                )

                row[f"{model.name}.grade"] = grading_result

        elif mode == "arena":
            model1, model2 = random.sample(models_to_eval, 2)

            model1_output = model1.get_generation(task)
            model2_output = model2.get_generation(task)

            row[f"{model1.name}.output"] = model1_output
            row[f"{model2.name}.output"] = model2_output

            arena_result = judging_model(
                task, success_criteria, failure_criteria, model1_output, model2_output
            )  # either 1 or 2 depending on which model won

            if arena_result == 1:
                # then we make it so that the model1 = 1 and model2 = -1
                row[f"{model1.name}.grade"] = 1
                row[f"{model2.name}.grade"] = -1

            elif arena_result == 2:
                row[f"{model1.name}.grade"] = -1
                row[f"{model2.name}.grade"] = 1

            else:
                # raise error
                raise ValueError(f"Invalid arena result: {arena_result}")

        # we then patch back the row
        tasks.loc[idx] = row

        # we need to then return the result

        results = []
        tasks.to_csv(processed_tasks_path, index=False)

        if mode == "single":
            # we can just take the sum of model_results per model
            """
                [
                    {
                        "model": model_name<str>, 
                        "score": score<int>
                    }
                ]
            """
            for model in models_to_eval:
                model_results = tasks[f"{model.name}.grade"]
                score = model_results.sum()
                results.append({"model": model.name, "score": score})

        elif mode == "arena":
            ratings = {model.name: 1000 for model in models_to_eval}
            k_factor = 32

            for idx, row in tasks.iterrows():
                # get non-zero columns that end with ".grade"
                non_zero_columns = [
                    column
                    for column in row.index
                    if column.endswith(".grade") and row[column] != 0
                ]

                if len(non_zero_columns) != 2:
                    raise ValueError(
                        f"Invalid number of non-zero columns: {len(non_zero_columns)}"
                    )

                model1, model2 = non_zero_columns
                # remove the grade
                model1 = model1[:-6]
                model2 = model2[:-6]

                model1_rating = ratings[model1]
                model2_rating = ratings[model2]

                model_1_expected_score = 1 / (
                    1 + 10 ** ((model2_rating - model1_rating) / 400)
                )

                model_2_expected_score = 1 / (
                    1 + 10 ** ((model1_rating - model2_rating) / 400)
                )

                model1_score = max(row[f"{model1}.grade"], 0)
                model2_score = max(row[f"{model2}.grade"], 0)

                model1_new_rating = model1_rating + k_factor * (
                    model1_score - model_1_expected_score
                )

                model2_new_rating = model2_rating + k_factor * (
                    model2_score - model_2_expected_score
                )

                ratings[model1] = model1_new_rating
                ratings[model2] = model2_new_rating

            for model in models_to_eval:
                model_results = tasks[f"{model.name}.grade"]
                results.append(
                    {
                        "model": model.name,
                        "score": model_results.sum(),
                        "elo": ratings[model.name],
                        "num_matches": np.count_nonzero(
                            model_results
                        ),  # number of nonzero elements
                    }
                )

        # write results
        with open(results_path, "w") as f:
            json.dump(results, f)

        return results


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Roll your own evals")

#     parser.add_argument("data", type=str, help="The path to your csv file with tasks")
#     parser.add_argument(
#         "--models",
#         required=True,
#         type=str,
#         nargs="+",
#         help="The list of models to evaluate",
#     )

#     args = parser.parse_args()
#     print("data", args.data)
#     print("models", args.models)

#     models_to_test = []
#     for model in args.models:
#         models_to_test.append(LanguageModel(model))
