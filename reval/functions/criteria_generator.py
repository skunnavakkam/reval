from reval.language_models import LanguageModel


def criteria_generator(
    task,
    success_criteria=None,
    failure_criteria=None,
    good_example=None,
    bad_example=None,
):
    # if success_crtiera and failure_criteria are provided then we don't need to do anything and we should warn the user
    if success_criteria is not None and failure_criteria is not None:
        return success_criteria, failure_criteria

    if (
        good_example is None
        and bad_example is None
        and success_criteria is None
        and failure_criteria is None
    ):
        query = f"""You are an AI assistant tasked with generating success and failure criteria for a given task. This will help in evaluating how well other models perform on the task. Here's what you need to do:

1. I will provide you with a task description.
2. Your job is to generate clear, specific, and measurable success and failure criteria for this task.
3. The criteria should be detailed enough to allow for objective evaluation of task performance.

Here is the task query:
<task>
{task}
</task>

To generate the success and failure criteria, follow these steps:

1. Carefully analyze the task description.
2. For success criteria:
   a. Identify the key elements that would constitute a successful completion of the task.
   b. Create 3-5 specific, measurable criteria that, if met, would indicate the task was performed successfully.
   c. Ensure these criteria are directly related to the task goals and objectives.

3. For failure criteria:
   a. Consider potential ways the task could be performed incorrectly or incompletely.
   b. Create 3-5 specific, measurable criteria that, if met, would indicate the task was not performed successfully.
   c. Ensure these criteria address common mistakes or misunderstandings related to the task.

4. Review your criteria to ensure they are clear, objective, and directly related to the task.

Present your results in the following format:

<output>
<success_criteria>
[Success Criteria]
</success_criteria>

<failure_criteria>
[Failure Criteria]
</failure_criteria>
</output>

Ensure that your criteria are specific, measurable, and directly related to the given task. Avoid vague or subjective statements. Each criterion should be a complete sentence that clearly describes a specific aspect of task performance."""

        raw_response = LanguageModel("gpt-4o-mini")(query)
        success_criteria = raw_response.split("<success_criteria>")[1].split(
            "</success_criteria>"
        )[0]
        failure_criteria = raw_response.split("<failure_criteria>")[1].split(
            "</failure_criteria>"
        )[0]

        return success_criteria, failure_criteria

    elif bad_example is None and success_criteria is None and failure_criteria is None:
        query = f"""You are an AI assistant tasked with generating success and failure criteria for a given task. This will help in evaluating how well other models perform on the task. You will be provided with a task description and a good example of completing the task.

Here is the task query:
<task>
{task}
</task>

Here is a good example of completing the task:
<example>
{good_example}
</example>

Your job is to generate clear, specific, and measurable success and failure criteria for this task. These criteria should help in objectively evaluating the performance of other models on this task.

To generate success criteria:
1. Analyze the task description and the provided example carefully.
2. Identify key elements that contribute to successfully completing the task.
3. Create 3-5 specific, measurable criteria that would indicate a high-quality response to the task.
4. Ensure that the criteria are directly related to the task objectives and cover different aspects of performance.

To generate failure criteria:
1. Consider potential ways the task could be misunderstood or poorly executed.
2. Identify common mistakes or shortcomings that would result in an inadequate response.
3. Create 3-5 specific, measurable criteria that would indicate a low-quality or incorrect response to the task.
4. Ensure that the failure criteria are distinct from simply not meeting the success criteria.

Present your response in the following format:

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

Ensure that each criterion is clear, specific, and can be objectively measured or observed in a response to the task."""

        raw_response = LanguageModel("gpt-4o-mini")(query)
        success_criteria = raw_response.split("<success_criteria>")[1].split(
            "</success_criteria>"
        )[0]
        failure_criteria = raw_response.split("<failure_criteria>")[1].split(
            "</failure_criteria>"
        )[0]

        return success_criteria, failure_criteria

    elif good_example is None and success_criteria is None and failure_criteria is None:
        query = f"""You are an AI assistant tasked with generating success and failure criteria for a given task. This will help in evaluating how well other models perform on the task. You will be provided with a task description and a bad example of completing the task.

Here is the task query:
<task>
{task}
</task>

Here is a bad example of completing the task:
<example>
{bad_example}
</example>

Your job is to generate clear, specific, and measurable success and failure criteria for this task. These criteria should help in objectively evaluating the performance of other models on this task.

To generate success criteria:
1. Analyze the task description and the provided example carefully.
2. Identify key elements that contribute to successfully completing the task.
3. Create 3-5 specific, measurable criteria that would indicate a high-quality response to the task.
4. Ensure that the criteria are directly related to the task objectives and cover different aspects of performance.

To generate failure criteria:
1. Consider potential ways the task could be misunderstood or poorly executed.
2. Identify common mistakes or shortcomings that would result in an inadequate response.
3. Create 3-5 specific, measurable criteria that would indicate a low-quality or incorrect response to the task.
4. Ensure that the failure criteria are distinct from simply not meeting the success criteria.

Present your response in the following format:

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

Ensure that each criterion is clear, specific, and can be objectively measured or observed in a response to the task."""

        raw_response = LanguageModel("gpt-4o-mini")(query)
        success_criteria = raw_response.split("<success_criteria>")[1].split(
            "</success_criteria>"
        )[0]
        failure_criteria = raw_response.split("<failure_criteria>")[1].split(
            "</failure_criteria>"
        )[0]

        return success_criteria, failure_criteria

    # we can just check for success and failure criteria now
    if success_criteria is not None:
        # failure criteria is None
        query = f"""You are an AI assistant tasked with generating failure criteria for a given task or prompt. These failure criteria will be used to evaluate how well other AI models perform when responding to the prompt. Your goal is to create a set of specific, measurable criteria that indicate when a response fails to meet the task's objectives or violates important principles.

Here is the task prompt:
<task_prompt>
{task}
</task_prompt>

And here are the success criteria for this task:
<success_criteria>
{success_criteria}
</success_criteria>

Your job is to generate a set of failure criteria that complement the success criteria. These failure criteria should identify specific ways in which a response could fall short of the task's requirements or expectations.

To generate effective failure criteria:

1. Carefully analyze the task prompt and success criteria.
2. Identify key elements, requirements, and objectives of the task.
3. Consider potential misunderstandings, misinterpretations, or mistakes that could lead to inadequate responses.
4. Think about ways in which a response could partially meet the criteria but still fall short of full success.
5. Consider ethical, logical, or practical issues that might arise in responses.

When formulating your failure criteria:

1. Make each criterion specific and measurable.
2. Ensure the criteria are directly related to the task and its objectives.
3. Cover a range of potential issues, from minor shortcomings to major failures.
4. Include criteria that address both the content and the format of the response, if applicable.
5. Consider including criteria related to adherence to instructions, relevance, coherence, and completeness.

Present your failure criteria in a numbered list, with each criterion clearly stated. 

Present your criteria in the following format:
<failure_criteria>
[failure_criteria]
</failure_criteria>
"""

        raw_response = LanguageModel("gpt-4o-mini")(query)
        failure_criteria = raw_response.split("<failure_criteria>")[1].split(
            "</failure_criteria>"
        )[0]
        return success_criteria, failure_criteria

    if failure_criteria is not None:
        query = f"""You are tasked with creating success criteria for a given task based on its prompt and failure criteria. These success criteria will be used to evaluate how well other AI models perform in response to the task prompt.

First, carefully review the following task prompt:

<task_prompt>
{task}
</task_prompt>

Now, consider the failure criteria for this task:

<failure_criteria>
{failure_criteria}
</failure_criteria>

Your goal is to create a set of success criteria that complement and expand upon the failure criteria. Success criteria should:

1. Be specific and measurable
2. Align with the objectives of the task prompt
3. Cover different aspects of performance (e.g., accuracy, completeness, creativity, adherence to instructions)
4. Be achievable and realistic
5. Provide a clear distinction between successful and unsuccessful responses

To create the success criteria:

1. Analyze the task prompt and identify key requirements and objectives
2. Consider the failure criteria and think about what the opposite of failure would look like
3. Break down the task into its core components and create criteria for each
4. Ensure that your criteria are comprehensive and cover all aspects of a successful response
5. Use clear and concise language to describe each criterion

Present your success criteria in a numbered list format. For each criterion, provide a clear description of the criterion and if applicable, a suggestion for how to measure or evaluate this criterion

Format your response as follows:

<success_criteria>
1. [Criterion 1]
2. [Criterion 2]

[Continue with additional criteria as needed]
</success_criteria>
"""

        raw_response = LanguageModel("gpt-4o-mini")(query)
        success_criteria = raw_response.split("<success_criteria>")[1].split(
            "</success_criteria>"
        )[0]
        return success_criteria, failure_criteria
