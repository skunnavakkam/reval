from reval.language_models import OpenAIModel, AsyncOpenAIModel
from string import Template


class CriteriaGenerator:
    def __init__(self):
        self.model = OpenAIModel("gpt-4o-mini", max_tokens=16384, temperature=0.5)
        self.prompt = Template(
            """You are an AI tasked with generating specific success criteria and failure criteria to grade another model's performance on a given task. Your goal is to create clear, measurable, and relevant criteria that can be used to evaluate the quality and effectiveness of the model's output.

Here is the task description:
<task>
$task
</task>

To generate success criteria:
1. Carefully analyze the task description and identify key elements that a successful completion would entail.
2. Create 3-5 specific, measurable criteria that would indicate a high-quality performance of the task.
3. Ensure each criterion is directly related to the task and can be objectively assessed.
4. Consider aspects such as accuracy, completeness, relevance, and adherence to any specified guidelines or constraints.

To generate failure criteria:
1. Identify potential pitfalls or common mistakes that could occur when attempting the task.
2. Create 3-5 specific, measurable criteria that would indicate a poor or inadequate performance of the task.
3. Ensure each criterion represents a clear failure to meet the task requirements or a significant error in execution.
4. Consider aspects such as inaccuracy, incompleteness, irrelevance, or violation of specified guidelines or constraints.

Present your criteria in the following format:

<criteria>
<success_criteria>
1. [First success criterion]
2. [Second success criterion]
3. [Third success criterion]
[Additional criteria if necessary]
</success_criteria>

<failure_criteria>
1. [First failure criterion]
2. [Second failure criterion]
3. [Third failure criterion]
[Additional criteria if necessary]
</failure_criteria>
</criteria>

Ensure that your criteria are specific to the given task, avoiding generic or vague statements. Each criterion should be clear, concise, and directly applicable to evaluating the model's performance on the task."""
        )

    def __call__(
        self,
        task,
        success_criteria,
        failure_criteria,
        correct_answer,
        good_example,
        bad_example,
    ):
        prompt = self.prompt.substitute(task=task)
        response = self.model(prompt)
        success = response.split("<success_criteria>")[1].split("</success_criteria>")[
            0
        ]
        failure = response.split("<failure_criteria>")[1].split("</failure_criteria>")[
            0
        ]
        return (task, success, failure, correct_answer, good_example, bad_example)


class AsyncCriteriaGenerator:
    def __init__(self):
        self.model = AsyncOpenAIModel("gpt-4o-mini", max_tokens=16384, temperature=0.5)
        self.prompt = Template(
            """You are an AI tasked with generating specific success criteria and failure criteria to grade another model's performance on a given task. Your goal is to create clear, measurable, and relevant criteria that can be used to evaluate the quality and effectiveness of the model's output.

Here is the task description:
<task>
$task
</task>

To generate success criteria:
1. Carefully analyze the task description and identify key elements that a successful completion would entail.
2. Create 3-5 specific, measurable criteria that would indicate a high-quality performance of the task.
3. Ensure each criterion is directly related to the task and can be objectively assessed.
4. Consider aspects such as accuracy, completeness, relevance, and adherence to any specified guidelines or constraints.

To generate failure criteria:
1. Identify potential pitfalls or common mistakes that could occur when attempting the task.
2. Create 3-5 specific, measurable criteria that would indicate a poor or inadequate performance of the task.
3. Ensure each criterion represents a clear failure to meet the task requirements or a significant error in execution.
4. Consider aspects such as inaccuracy, incompleteness, irrelevance, or violation of specified guidelines or constraints.

Present your criteria in the following format:

<criteria>
<success_criteria>
1. [First success criterion]
2. [Second success criterion]
3. [Third success criterion]
[Additional criteria if necessary]
</success_criteria>

<failure_criteria>
1. [First failure criterion]
2. [Second failure criterion]
3. [Third failure criterion]
[Additional criteria if necessary]
</failure_criteria>
</criteria>

Ensure that your criteria are specific to the given task, avoiding generic or vague statements. Each criterion should be clear, concise, and directly applicable to evaluating the model's performance on the task."""
        )

    async def __call__(
        self,
        task,
        success_criteria,
        failure_criteria,
        correct_answer,
        good_example,
        bad_example,
    ):
        prompt = self.prompt.substitute(task=task)
        response = await self.model(prompt)
        success = response.split("<success_criteria>")[1].split("</success_criteria>")[
            0
        ]
        failure = response.split("<failure_criteria>")[1].split("</failure_criteria>")[
            0
        ]
        return (task, success, failure, correct_answer, good_example, bad_example)
