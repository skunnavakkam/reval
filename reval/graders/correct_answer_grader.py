from reval.language_models import GenericLanguageModel, AsyncGenericLanguageModel
from string import Template


class CorrectAnswerGrader:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model = GenericLanguageModel(model_name)
        self.prompt = Template(
            """You are tasked with checking if a given response contains a correct answer to a task. You will be provided with three pieces of information:

<task>
$task
</task>

<response>
$model_response
</response>

<correct_answer>
$correct_answer
</correct_answer>

Your job is to determine whether the response contains a correct answer to the question. Reason through it, and then return your final answer.

<assessment>
[CORRECT] or INCORRECT
</assessment>

Remember, your goal is to determine if the response contains a correct answer, not to judge the quality or completeness of the explanation beyond that."""
        )
        pass

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

        for model_response in model_responses:
            prompt = self.prompt.substitute(
                task=task,
                model_response=model_response,
                correct_answer=correct_answer,
            )
            response = self.model(prompt)
            grade = response.split("<assessment>")[1].split("</assessment>")[0]
            ret.append(int("correct" in grade.lower()))

        return ret


class AsyncCorrectAnswerGrader:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model = AsyncGenericLanguageModel(model_name)
        self.prompt = Template(
            """You are tasked with checking if a given response contains a correct answer to a task. You will be provided with three pieces of information:

<task>
$task
</task>

<response>
$model_response
</response>

<correct_answer>
$correct_answer
</correct_answer>

Your job is to determine whether the response contains a correct answer to the question. Reason through it, and then return your final answer.

<assessment>
[CORRECT] or INCORRECT
</assessment>

Remember, your goal is to determine if the response contains a correct answer, not to judge the quality or completeness of the explanation beyond that."""
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

        for model_response in model_responses:
            prompt = self.prompt.substitute(
                task=task,
                model_response=model_response,
                correct_answer=correct_answer,
            )
            response = await self.model(prompt)
            grade = response.split("<assessment>")[1].split("</assessment>")[0]
            ret.append(int("correct" in grade.lower()))

        return ret
