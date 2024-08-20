import pandas as pd
from typing import List, Callable, Any
import asyncio
from aiolimiter import AsyncLimiter


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
    df = df.copy()

    # create a grades dataframe
    grades_df = pd.DataFrame(
        columns=[
            model.name if hasattr(model, "name") else model.__name__
            for model in models_to_judge
        ]
    )

    for idx, row in df.iterrows():
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

        for model, grade in zip(models_to_judge, grades):
            name = model.name if hasattr(model, "name") else model.__name__
            grades_df.loc[idx].loc[name] = grade

        # include the task data in the original dataframe
        for i, col in enumerate(
            [
                "tasks",
                "success criteria",
                "failure criteria",
                "correct answer",
                "good example",
                "bad example",
            ]
        ):
            if col in df.columns:
                df.loc[idx].loc[col] = task_data[i]

    # concatenate and return
    return pd.concat([df, grades_df], axis=1)


def is_async_function(func: Callable) -> bool:
    return asyncio.iscoroutinefunction(func)


async def run_async(func: Callable, *args, **kwargs) -> Any:
    if is_async_function(func):
        return await func(*args, **kwargs)
    else:
        return await asyncio.to_thread(func, *args, **kwargs)


async def AsyncReval(
    df: pd.DataFrame,
    models_to_judge: List[Callable],
    preprocessing: List[Callable],
    grader: Callable,
    rate_limit: float = None,
) -> pd.DataFrame:
    col = [c.lower() for c in df.columns]
    if "tasks" not in col:
        raise ValueError("Tasks column not found in the dataframe")

    df = df.copy()

    grades_df = pd.DataFrame(
        columns=[
            model.name if hasattr(model, "name") else model.__name__
            for model in models_to_judge
        ]
    )

    limiter = AsyncLimiter(rate_limit) if rate_limit else None

    async def process_row(idx: int, row: pd.Series) -> None:
        task_data = (
            row["tasks"],
            row.get("success criteria", None),
            row.get("failure criteria", None),
            row.get("correct answer", None),
            row.get("good example", None),
            row.get("bad example", None),
        )

        for step in preprocessing:
            task_data = await run_async(step, *task_data)

        async def get_model_response(model: Callable) -> Any:
            if limiter:
                async with limiter:
                    return await run_async(model, task_data[0])
            else:
                return await run_async(model, task_data[0])

        model_responses = await asyncio.gather(
            *[get_model_response(model) for model in models_to_judge]
        )

        grades = await run_async(grader, *task_data, model_responses)

        for model, grade in zip(models_to_judge, grades):
            name = model.name if hasattr(model, "name") else model.__name__
            grades_df.loc[idx, name] = grade

        for i, col in enumerate(
            [
                "tasks",
                "success criteria",
                "failure criteria",
                "correct answer",
                "good example",
                "bad example",
            ]
        ):
            if col in df.columns:
                df.loc[idx, col] = task_data[i]

    await asyncio.gather(*[process_row(idx, row) for idx, row in df.iterrows()])

    return pd.concat([df, grades_df], axis=1)
