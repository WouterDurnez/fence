from fence import LLM, Link, PromptTemplate
import re
from fence.demo.demo_label.prompt_templates import (
    POLICY_TEMPLATE,

)
from fence.src.llm.parsers import TOMLParser, TripleBacktickParser

LINK_TEMPLATES = {
    "policy": POLICY_TEMPLATE,
}
LINK_PARSERS = {
    "policy": TripleBacktickParser(),
}



def build_links( llm: LLM) -> list[Link]:
    """
    Build and return a list of links based on the provided recipe and templates.
    :param llm: LLM instance
    """

    link = Link(
        name=f"policy_link",
        template=PromptTemplate(
            template=POLICY_TEMPLATE, input_variables=["state", "recipe"]
        ),
        llm=llm,
        parser=TripleBacktickParser(),
        output_key=f"policy_output",
            )

    return [link]

class FilenameProcessor:
    """
    A class to process filenames.
    """
    CASE_MAP = {
        "title": str.title,
        "upper": str.upper,
        "lower": str.lower,
    }

    def __init__(self, capitalisation: str = None, separator: str = None, remove_special_characters: bool = None,
                 date: str = None, date_location: str = "end", truncation_limit: int = 50):
        """
        Initialize the FilenameProcessor with the given parameters.
        """
        self.capitalisation = capitalisation
        self.separator = separator if separator else ""
        self.date = date
        self.date_location = date_location
        self.truncation_limit = truncation_limit
        self.remove_special_characters = remove_special_characters

        # Regex patterns
        self._special_character_pattern = re.compile(r"[^a-zA-Z0-9-_]+")
        self._camel_case_pattern = re.compile("([a-z])([A-Z])")
        self._separator_pattern = re.compile("[_\\-./]")

    def _remove_special_characters(self):
        """
        Replace special characters with a single separator.
        """
        self.filename = self._special_character_pattern.sub(self.separator, self.filename)

    def _split_camel_case(self):
        """
        Split the filename by camel case.
        """
        self.filename = self._camel_case_pattern.sub(r"\1 \2", self.filename)

    def _split_by_separators(self):
        """
        Split the filename by separators.
        """
        self.filename = self._separator_pattern.sub(lambda x: " ", self.filename)

    def _split_by_spaces(self):
        """
        Split the filename by spaces.
        """
        self.filename = re.sub(" +", " ", self.filename)

    def _remove_spaces(self):
        """
        Remove leading and trailing spaces and convert the filename to lowercase.
        """
        self.filename = self.filename.strip()


    def _split_into_components(self):
        """
        Split the filename into components.
        """
        self.filename_components = self.filename.split(" ")

    def _convert_to_desired_case(self):
        """
        Convert the filename to the desired case.
        """
        self.filename_components = [self.CASE_MAP[self.capitalisation](component.lower()) for component in self.filename_components]


    def _join_components(self):
        """
        Join the components of the filename with the separator.
        """
        self.filename = self.separator.join(self.filename_components)

    def _truncate_and_add_date(self):
        """
        Truncate the filename if it is too long and add the date.
        """
        if self.date:
            file_name = self.filename[: self.truncation_limit - len(self.date) - 1]

            if self.date_location == "end":
                self.filename = f"{file_name}_{self.date}"
            else:
                self.filename = f"{self.date}_{file_name}"
        else:
            self.filename = self.filename[:self.truncation_limit]

    def _finish_up(self):
        """
        If the final character is a separator, remove it.
        """
        if self.filename[-1] in self.separator:
            self.filename = self.filename[:-1]

    def process_filename(self, filename:str):
        """
        Process the given filename by calling all the other methods in the correct order.

        :return: The processed filename
        """
        self.filename = filename
        self._split_camel_case()
        self._split_by_separators()
        self._split_by_spaces()
        self._remove_spaces()


        # String to list #

        self._split_into_components()
        if self.capitalisation:
            self._convert_to_desired_case()

        # List to string #

        self._join_components()

        # Date, truncation, special characters #

        self._truncate_and_add_date()
        if self.remove_special_characters:
            self._remove_special_characters() # Potentially: Replace special characters with SINGLE separator
        self._finish_up()
        return self.filename

if __name__ == '__main__':

    filenames = [
        "example.SomeReallyLongFILENAME also-Separators/OrCamelCase_orBothANDNumbers1234567890.did-i-mention-it-was-long",
        "someNewFiles_DK",
        "anotherFile.v1",
        "file1",
        "file2.loadsofnumbers1234567890",
        "filen@#ame_with_sp*cial_characters!@#$%^&*()_+",

    ]

    processor = FilenameProcessor()
    processor_with_prefixed_date = FilenameProcessor(date="2021-01-01", date_location="start")
    processor_with_suffixed_date_and_title = FilenameProcessor(date="2021-01-01", date_location="end", capitalisation="title")
    processor_with_uppercase = FilenameProcessor(capitalisation="upper")
    processor_with_lower_limit = FilenameProcessor(truncation_limit=10)
    processor_no_special_characters = FilenameProcessor(remove_special_characters=True)

    for filename in filenames:
        print(f"FILENAME: {filename}")
        print(f" --> processed: {processor.process_filename(filename)}")
        print(f" --> processed with prefixed date: {processor_with_prefixed_date.process_filename(filename)}")
        print(f" --> processed with suffixed date and title: {processor_with_suffixed_date_and_title.process_filename(filename)}")
        print(f" --> processed with uppercase: {processor_with_uppercase.process_filename(filename)}")
        print(f" --> processed with lower limit: {processor_with_lower_limit.process_filename(filename)}")
        print(f" --> processed with no special characters: {processor_no_special_characters.process_filename(filename)}")
        print()
