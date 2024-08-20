import pandas as pd
from reval import Reval
import argparse


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
