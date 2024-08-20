from reval.language_models import GenericLanguageModel, AsyncGenericLanguageModel
from string import Template


class CriteriaBinaryGrader:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model = GenericLanguageModel(model_name)
        self.prompt = Template(
            """You are tasked with grading a model's response to a given task. Your job is to carefully evaluate the response against the provided success and failure criteria, and assign a grade based on the given grading scale. Follow these steps:

1. Review the following information:

<task>
$task
</task>

$success_criteria

$failure_criteria

2. Now, examine the model's response:

<model_response>
$model_response
</model_response>

Analyze the model's response thoroughly. Compare it against both the success and failure criteria. Consider how well the response meets the requirements of the task and how it aligns with or deviates from the given criteria. Provide a detailed justification for your grading decision. Include specific examples from the model's response that support your evaluation. 

Finally, return your grade, either 1 if the model was successful or 0 if the model was unsuccessful, using

<grade>
[Return your final grade here]
</grade."""
        )

    def __call__(
        self,
        task,
        success_criteria,
        failure_criteria,
        correct_answer,
        good_example,
        bad_example,
        model_responses,
    ):
        ret = []

        success_criteria = (
            "<success_criteria>" + success_criteria + "</success_criteria>"
            if success_criteria is not None
            else ""
        )
        failure_criteria = (
            "<failure_criteria>" + failure_criteria + "</failure_criteria>"
        )

        for model_response in model_responses:
            prompt = self.prompt.substitute(
                task=task,
                success_criteria=success_criteria,
                failure_criteria=failure_criteria,
                model_response=model_response,
            )
            response = self.model(prompt)
            grade = response.split("<grade>")[1].split("</grade>")[0]
            grade = int(grade)
            ret.append(grade)

        return ret


class AsyncCriteriaBinaryGrader:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model = AsyncGenericLanguageModel(model_name)
        self.prompt = Template(
            """You are tasked with grading a model's response to a given task. Your job is to carefully evaluate the response against the provided success and failure criteria, and assign a grade based on the given grading scale. Follow these steps:

1. Review the following information:

<task>
$task
</task>

$success_criteria

$failure_criteria

2. Now, examine the model's response:

<model_response>
$model_response
</model_response>

Analyze the model's response thoroughly. Compare it against both the success and failure criteria. Consider how well the response meets the requirements of the task and how it aligns with or deviates from the given criteria. Provide a detailed justification for your grading decision. Include specific examples from the model's response that support your evaluation. 

Finally, return your grade, either 1 if the model was successful or 0 if the model was unsuccessful, using

<grade>
[Return your final grade here]
</grade."""
        )

    async def __call__(
        self,
        task,
        success_criteria,
        failure_criteria,
        correct_answer,
        good_example,
        bad_example,
        model_responses,
    ):
        ret = []

        success_criteria = (
            "<success_criteria>" + success_criteria + "</success_criteria>"
            if success_criteria is not None
            else ""
        )
        failure_criteria = (
            "<failure_criteria>" + failure_criteria + "</failure_criteria>"
        )

        for model_response in model_responses:
            prompt = self.prompt.substitute(
                task=task,
                success_criteria=success_criteria,
                failure_criteria=failure_criteria,
                model_response=model_response,
            )
            response = await self.model(prompt)
            grade = response.split("<grade>")[1].split("</grade>")[0]
            grade = int(grade)
            ret.append(grade)

        return ret


class CriteriaScaleGrader:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model = GenericLanguageModel(model_name)
        self.prompt = Template(
            """You are tasked with grading a model's response to a given task. Your job is to carefully evaluate the response against the provided success and failure criteria, and assign a grade based on the given grading scale. Follow these steps:

1. Review the following information:

<task>
$task
</task>

$success_criteria

$failure_criteria

2. Now, examine the model's response:

<model_response>
$model_response
</model_response>

Analyze the model's response thoroughly. Compare it against both the success and failure criteria. Consider how well the response meets the requirements of the task and how it aligns with or deviates from the given criteria. Provide a detailed justification for your grading decision. Include specific examples from the model's response that support your evaluation. 

Finally, return your grade, which should be a number between 0 and 100

<grade>
[Return your final grade here]
</grade."""
        )

    def __call__(
        self,
        task,
        success_criteria,
        failure_criteria,
        correct_answer,
        good_example,
        bad_example,
        model_responses,
    ):
        ret = []

        success_criteria = (
            "<success_criteria>" + success_criteria + "</success_criteria>"
            if success_criteria is not None
            else ""
        )
        failure_criteria = (
            "<failure_criteria>" + failure_criteria + "</failure_criteria>"
        )

        for model_response in model_responses:
            prompt = self.prompt.substitute(
                task=task,
                success_criteria=success_criteria,
                failure_criteria=failure_criteria,
                model_response=model_response,
            )
            response = self.model(prompt)
            grade = response.split("<grade>")[1].split("</grade>")[0]
            grade = int(grade) / 100
            ret.append(grade)

        return ret


class AsyncCriteriaScaleGrader:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model = AsyncGenericLanguageModel(model_name)
        self.prompt = Template(
            """You are tasked with grading a model's response to a given task. Your job is to carefully evaluate the response against the provided success and failure criteria, and assign a grade based on the given grading scale. Follow these steps:

1. Review the following information:

<task>
$task
</task>

$success_criteria

$failure_criteria

2. Now, examine the model's response:

<model_response>
$model_response
</model_response>

Analyze the model's response thoroughly. Compare it against both the success and failure criteria. Consider how well the response meets the requirements of the task and how it aligns with or deviates from the given criteria. Provide a detailed justification for your grading decision. Include specific examples from the model's response that support your evaluation. 

Finally, return your grade, which should be a number between 0 and 100

<grade>
[Return your final grade here]
</grade."""
        )

    async def __call__(
        self,
        task,
        success_criteria,
        failure_criteria,
        correct_answer,
        good_example,
        bad_example,
        model_responses,
    ):
        ret = []

        success_criteria = (
            "<success_criteria>" + success_criteria + "</success_criteria>"
            if success_criteria is not None
            else ""
        )
        failure_criteria = (
            "<failure_criteria>" + failure_criteria + "</failure_criteria>"
        )

        for model_response in model_responses:
            prompt = self.prompt.substitute(
                task=task,
                success_criteria=success_criteria,
                failure_criteria=failure_criteria,
                model_response=model_response,
            )
            response = await self.model(prompt)
            grade = response.split("<grade>")[1].split("</grade>")[0]
            grade = int(grade) / 100
            ret.append(grade)

        return ret
