# Code Structure

## Data layer 

### General Knowledge and Reasoning Datasets 

These datasets are used to evaluate the general knowledge and reasoning capabilities of LLMs. They typically contain a list of questions, and each question has a list of choices and a correct answer. 

Currently, we support the following datasets:
- MMLU-Pro
- AIME 2025
- GPQA (main, diamond, extended, experts)

### Dataset 

A Dataset is a structured object that contains the following fields:

Required fields:
- name (str): The name of the dataset.
- questions (list[Question]): The list of questions in the dataset.
- prompts (dict[str, str]): A dictionary of default prompts for the dataset. The key is the prompt type, and the value is the prompt text. This is used because many benchmark datasets come with their default prompts so people can compare the performance of different LLMs on the same prompt. The prompt type is usually "zero_shot", "chain_of_thought", or "five_shot", etc. 

Optional fields:
- description (str): The description of the dataset.
- metadata (dict[str, any]): The metadata of the dataset. This is used to store additional information about the dataset.
- categories (list[str]): The list of categories in the dataset.
- questions_by_category (dict[str, list[Question]]): A dictionary of questions by category. The key is the category, and the value is the list of questions in the category.


### Question 

A Question is a structured object that contains the following fields:

Required fields:
- id (int): The unique identifier of the question. This is used to identify the question in the dataset.
- original_id (str): The original identifier of the question from the original dataset.
- question (str): The text of the question.
- choices (dict[str, str]): A dictionary of choices for the question. The key is the choice letter, and the value is the choice text. We use strings to index the choices, e.g. "A", "B", "C", "D". The order of the choices is important, and we use the order to map the answer to the correct choice.
- answer (str): The correct answer to the question. We use the string to index the answer, e.g. "A", "B", "C", "D".

Optional fields:
- category (str): The category (sometimes called domain, or subject) of the question. This is used to group the question with other questions of the same category. For example, in MMLU-Pro, the category is the subject of the question. In AIME 2025, the category is the problem type. 
- subcategory (str): The sub-category of the question. This is used to group the question with other questions of the same sub-category. 
- difficulty (str): The difficulty of the question. This is used to group the question with other questions of the same difficulty.
- tags (list[str]): The tags of the question. This is used to group the question with other questions of the same tags.
- metadata (dict[str, any]): The metadata of the question. This is used to store additional information about the question.