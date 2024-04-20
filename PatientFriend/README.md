# ChatPatient API

## Overview
The `ChatPatient` API leverages a Flask-based web service to provide two main functionalities:
1. **Summarization of Radiology Reports**: Converts detailed radiology text into simplified, patient-friendly summaries.
2. **Translation of Text**: Translates text from English to other languages, defaulting to Hindi but capable of handling other languages as specified.

## Installation

### Prerequisites
- Python 3.6 or higher.
- Flask
- An instance of `ChatPatient` from the `ChatPatient.py` module, which utilizes OpenAI's GPT-4 model.

### Dependencies
Before running the API, you will need to install the necessary Python packages. Run the following command to install Flask:

```bash
pip install flask
```

### API Endpoints

#### Summarize Text
URL: /api/summarize
Method: POST
```
{
  "text": "[plain text string of the radiology report]"
}
```

Success Response:
Code: 200
Content:

```
{
  "summary": "[summary of the report]"
}
```

Error Response:
Code: 400 BAD REQUEST
Content: {"error": "No text provided"}
Code: 500 INTERNAL SERVER ERROR
Content: {"error": "[error message]"}

#### Translate Text
URL: /api/translate
Method: POST
Data Constraints:
json
```{
  "text": "[plain text string]",
  "language": "[optional; default is 'hindi']"
}
```
Success Response:
Code: 200
Content:
json
```{
  "translated_text": "[translated text]"
}
```

Error Response:
Code: 400 BAD REQUEST
Content: {"error": "No text provided"}
Code: 500 INTERNAL SERVER ERROR
Content: {"error": "[error message]"}


### Usage Examples
To interact with the API, you can use tools like curl or Postman. Here are some example curl commands:

Summarize Text
```
bash
curl -X POST http://localhost:5000/api/summarize -H "Content-Type: application/json" -d '{"text":"Patient shows signs of consolidation..."}'
```
Translate Text
```
bash
Copy code
curl -X POST http://localhost:5000/api/translate -H "Content-Type: application/json" -d 
```

# ChatPatient

## Overview
The `ChatPatient` class leverages OpenAI's powerful GPT-4 model to provide patient-friendly summaries of complex medical texts, specifically radiology reports. This tool is designed to make radiology reports accessible at an 8th-grade reading level. Additionally, the class offers functionality to translate text from English to other languages, with default support for Hindi.

## Features
- **Patient-Friendly Summaries**: Simplify radiology reports into clear, concise text that is easy to understand for non-medical readers.
- **Language Translation**: Translate text from English to a specified language, supporting diverse linguistic needs.

## Installation

### Prerequisites
- Python 3.6 or newer.
- Access to OpenAI's API with credentials.

### Libraries
This project requires the following Python libraries:
- `dotenv`: To manage environment variables.
- `langchain_openai`: To interact with OpenAI's GPT model.

Install these libraries using pip:

```bash
pip install python-dotenv langchain_openai
```

### Usage
#### Creating an Instance of ChatPatient
First, initialize the ChatPatient:
```from ChatPatient import ChatPatient
chat_patient = ChatPatient()
```
#### Generating Patient-Friendly Summaries
To generate a summary from a radiology report:
```report_text = "Detailed description of the radiology report..."
summary = chat_patient.get_friendly_text(report_text)
print(summary)
```

#### Translating Text
To translate text to another language:
 
```english_text = "Good morning, how can I help you today?"
translated_text = chat_patient.translate_text(english_text, language="hindi")
print(translated_text)
```

#### Error Handling
The class includes basic error handling for common issues, such as API errors:
```try:
    summary = chat_patient.get_friendly_text("Radiology report text here...")
except Exception as e:
    print(f"An error occurred: {e}")

```

