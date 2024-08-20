from reval.language_models import OpenAIModel, AsyncOpenAIModel
from string import Template


class CriteriaGeneratorWithExamples:
    def __init__(self):
        self.model = OpenAIModel("gpt-4o-mini", max_tokens=16384, temperature=0.5)
        self.prompt = Template(
            """You are an AI tasked with generating success and failure criteria for evaluating another model's performance on a given task. Your goal is to create a comprehensive set of criteria that can be used to grade the model's output effectively.

Here is the task description:
<task_description>
$task
</task_description>

Here are the examples provided (if any):
<good_example>
$good_example
</good_example>

<bad_example>
$bad_example
</bad_example>

Carefully analyze the task description and the provided examples (if any). Consider the following aspects:
1. The main objectives of the task
2. Key elements that should be present in a successful response
3. Common pitfalls or errors that might lead to failure
4. Any specific requirements or constraints mentioned

Based on your analysis, generate a set of success criteria and failure criteria. These criteria should be specific, measurable, and directly related to the task at hand.

First, generate the success criteria. These should outline what constitutes a high-quality, accurate, and complete response to the task. Consider including criteria related to:
- Accuracy of information
- Completeness of the response
- Adherence to any specified formats or guidelines
- Relevance to the task
- Quality of reasoning or analysis (if applicable)

Next, generate the failure criteria. These should describe what would make a response inadequate, incorrect, or incomplete. Consider including criteria related to:
- Factual errors or inaccuracies
- Incomplete or missing information
- Violation of specified guidelines or constraints
- Irrelevance to the task
- Poor reasoning or analysis (if applicable)

Present your criteria in the following format:

<success_criteria>
1. [First success criterion]
2. [Second success criterion]
3. [Third success criterion]
...
</success_criteria>

<failure_criteria>
1. [First failure criterion]
2. [Second failure criterion]
3. [Third failure criterion]
...
</failure_criteria>

Ensure that your criteria are clear, concise, and directly applicable to the task. They should be specific enough to guide evaluation but general enough to cover various possible responses."""
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
        prompt = self.prompt.substitute(
            task=task,
            good_example=good_example if good_example is not None else "None",
            bad_example=bad_example if bad_example is not None else "None",
        )
        response = self.model(prompt)
        success = response.split("<success_criteria>")[1].split("</success_criteria>")[
            0
        ]
        failure = response.split("<failure_criteria>")[1].split("</failure_criteria>")[
            0
        ]
        return (task, success, failure, correct_answer, good_example, bad_example)


class AsyncCriteriaGeneratorWithExamples:
    def __init__(self):
        self.model = AsyncOpenAIModel("gpt-4o-mini", max_tokens=16384, temperature=0.5)
        self.prompt = Template(
            """You are an AI tasked with generating success and failure criteria for evaluating another model's performance on a given task. Your goal is to create a comprehensive set of criteria that can be used to grade the model's output effectively.

Here is the task description:
<task_description>
$task
</task_description>

Here are the examples provided (if any):
<good_example>
$good_example
</good_example>

<bad_example>
$bad_example
</bad_example>

Carefully analyze the task description and the provided examples (if any). Consider the following aspects:
1. The main objectives of the task
2. Key elements that should be present in a successful response
3. Common pitfalls or errors that might lead to failure
4. Any specific requirements or constraints mentioned

Based on your analysis, generate a set of success criteria and failure criteria. These criteria should be specific, measurable, and directly related to the task at hand.

First, generate the success criteria. These should outline what constitutes a high-quality, accurate, and complete response to the task. Consider including criteria related to:
- Accuracy of information
- Completeness of the response
- Adherence to any specified formats or guidelines
- Relevance to the task
- Quality of reasoning or analysis (if applicable)

Next, generate the failure criteria. These should describe what would make a response inadequate, incorrect, or incomplete. Consider including criteria related to:
- Factual errors or inaccuracies
- Incomplete or missing information
- Violation of specified guidelines or constraints
- Irrelevance to the task
- Poor reasoning or analysis (if applicable)

Present your criteria in the following format:

<success_criteria>
1. [First success criterion]
2. [Second success criterion]
3. [Third success criterion]
...
</success_criteria>

<failure_criteria>
1. [First failure criterion]
2. [Second failure criterion]
3. [Third failure criterion]
...
</failure_criteria>

Ensure that your criteria are clear, concise, and directly applicable to the task. They should be specific enough to guide evaluation but general enough to cover various possible responses."""
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
        prompt = self.prompt.substitute(
            task=task,
            good_example=good_example if good_example is not None else "None",
            bad_example=bad_example if bad_example is not None else "None",
        )
        response = await self.model(prompt)
        success = response.split("<success_criteria>")[1].split("</success_criteria")[0]
        failure = response.split("<failure_criteria>")[1].split("</failure_criteria")[0]
        return (task, success, failure, correct_answer, good_example, bad_example)
