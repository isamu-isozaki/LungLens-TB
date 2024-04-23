class ChatPatient:
    """
    A class to interact with an OpenAI Chat model to convert radiology report text
    into an 8th-grade level, patient-friendly summary.
    """
    def __init__(self) -> None:
        """
        Initializes the ChatPatient class by loading environment variables and setting up the chat model.
        """
        from dotenv import load_dotenv, find_dotenv
        from langchain_openai import ChatOpenAI

        # Load environment variables from .env file
        load_dotenv(find_dotenv())

        # Model configuration
        self.gpt_model = "gpt-4-1106-preview"
        self.llm = ChatOpenAI(model_name=self.gpt_model, temperature=0.3)

    def get_friendly_text(self, text: str) -> str:
        """
        Generates a patient-friendly summary of a given radiology report.

        Parameters:
        text (str): The radiology report text to summarize.

        Returns:
        str: An 8th-grade level summary of the radiology report.
        """

        prompt = "Given the following radiology report text, provide a paragraph summarizing what the report says in an 8th-grade level and patient-friendly manner."
        
        messages = [
            ("system", prompt),
            ("human", text),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content.replace("Summary","")
        except Exception as e:
            print(f"An error occurred: {e}")
            return "An error occurred while generating the summary. Please try again later."
        
    
    def translate_text(self, text, language="hindi"):
        """Translate the text based on the language passed.

        Args:
            text (_type_): Text to be translated
            language (str, optional): Defaults to "hindi".
        """
        messages = [
        ("system", f"You are a helpful assistant that translates English to {language}."),
        ("human", f"Translate this sentence from English to {language}. {text}"),
    ]


        return self.llm.invoke(messages).content



# Usage example:
# chat_patient = ChatPatient()
# summary = chat_patient.get_friendly_text("Here is the radiology report text...")
# print(summary)
