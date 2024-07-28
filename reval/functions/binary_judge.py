from reval.language_models import LanguageModel


def binary_judge(task, success_criteria, failure_criteria, model_output):
    query = f"""You are tasked with grading a language model's output based on a given prompt, success criteria, and failure criteria. Your goal is to provide an objective assessment of how well the model's output meets the specified criteria.

First, review the following information:

Prompt:
<prompt>
{task}
</prompt>

Model Output:
<model_output>
{model_output}
</model_output>

Success Criteria:
<success_criteria>
{success_criteria}
</success_criteria>

Failure Criteria:
<failure_criteria>
{failure_criteria}
</failure_criteria>

Carefully analyze the model's output in relation to the prompt, success criteria, and failure criteria. Consider how well the output addresses the prompt and meets the success criteria, as well as whether it exhibits any of the failure criteria.

Provide your reasoning for the grade you will assign. Consider both the strengths and weaknesses of the output, and explain how it aligns with or deviates from the given criteria. Be specific and reference parts of the output, success criteria, and failure criteria in your explanation.

After providing your reasoning, assign a binary grade, either 0 if the task was unsuccessful or 1 if the task was successful.

Present your evaluation in the following format:

<evaluation>
<reasoning>
[Your detailed reasoning for the grade]
</reasoning>
<grade>
[Numerical grade, either 0 or 1]
</grade>
</evaluation>

Ensure that your evaluation is impartial, thorough, and based solely on the provided information and criteria."""

    raw_response = LanguageModel("gpt-4o-mini").get_generation(query)
    score = int(raw_response.split("<grade>")[1].split("</grade>")[0])

    return score
