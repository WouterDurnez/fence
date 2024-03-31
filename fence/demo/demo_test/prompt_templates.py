# Add parent path to sys.path to import from src
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
print(sys.path)

from fence.src.llm.templates.string import StringTemplate

TEST_TEMPLATE = """
            You are an instructor in charge of creating exam questions.
            
            I will give you a piece of text, and I would like you to create an exam question based on it.

            The exam question is open ended. It is followed by multiple choice answer options with four options.

            The output should be in a single line TOML format, similar to the following example: 

            ```
            question = *your generated question, which is fully self-contained and has no references to the text*

            [[responses]]
            text = "response option 1"
            correct = true
            reason = *explain briefly why this is the correct response here*

            [[responses]]
            text = "response option 2"
            correct = false
            reason = *explain briefly why this is not the correct response here*

            [[responses]]
            text = "response option 3"
            correct = false
            reason = *explain briefly why this is not the correct response here*

            [[responses]]
            text = "response option 4"
            correct = false
            reason = *explain briefly why this is not the correct response here*
            ```

            This is the text:

            ```{{ highlight }}```

            Guidelines:
            - You are ONLY allowed to add valid TOML wrapped in triple backticks.
            - Make sure to mark the correct response in the TOML using booleans. Only one answer can be correct.
            - Always start your questions with "According to the passage, "!
            - Always add valid response options: "Don't know" is NEVER a valid response. You WILL be penalized for including this in the response options.
            - You must wrap all values in triple quotes, even if they are single words or numbers. This helps to ensure that the output is valid TOML, and that the parser can handle the output. 
            """

test_template = StringTemplate(source=TEST_TEMPLATE)

VERIFICATION_TEMPLATE = """
            You are an instructor in charge of verifying the quality of exam questions.
            
            Below, a piece of text is presented, followed by a JSON-formatted multiple choice question. Both are delimited by triple backticks.

            Can you please select the correct response in the JSON based on the information presented in the text? The correct response
            should be an integer that represents the index of the correct response in the JSON (starting from 0).

            "Don't know" is never a valid response, so never select that option should it be presented.

            Text:
            ```{{ highlight }}```

            Multiple choice question in JSON:
            ```{{ question_stripped }}```

            Response:

            Out of the options provided, the correct response index is (integer only):
            """

verification_template = StringTemplate(
    source=VERIFICATION_TEMPLATE
)

if __name__ == "__main__":




    # Test the prompt templates
    print(test_template(highlight="This is a test question."))
    print(
        verification_template(
            highlight="This is a test question.",
            question_stripped="This is a test question.",
        )
    )
