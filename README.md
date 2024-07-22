# custom_embedding_AI_chatbot


[Click here to watch the demo video](custom_chatbot_running.mp4)

https://github.com/user-attachments/assets/caf8ab5a-f84f-49e9-8f76-229a6d38403e


## Introduction
The custom_embedding_AI_chatbot App is a Python application designed for interacting with multiple PDF documents. By using natural language, you can ask questions about the content of the PDFs, and the app will provide relevant responses based on the documents' information. This application leverages a language model to deliver accurate answers to your queries. Please note that the app only responds to questions related to the loaded PDFs.

## Installation

To install the App, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```

3. Obtain an API key from Hugging face hub and add it to the `.env` file in the project directory.
  ```
  HUGGINGFACEHUB_API_TOKEN="api secret key"
  ```

4. Run the main python file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run custom_chatbot.py
   ```
5. The application will launch in your default web browser, displaying the user interface.

6. Load PDF documents into the app by following the provided instructions.

7. Ask questions in natural language about the loaded PDFs using the chat interface.

## Tweaking AI models

1. The default embedding model is "hkunlp/instructor-xl" which runs locally,
if you want to change it change the model

2. The default text llm model is "meta-llama/Meta-Llama-3-8B-Instruct",
if you want to change it ,replace the model_id-like "google/gemma-2-9b-it"

3. To set your custom instructions append more text in the instruction global variable

4. If you want to use OpenAI embedding model or GPT-models change the default model names to openAI models and add  it to the `.env` file in the project directory.
```
OPENAI_API_KEY="your_secret_api_key"
 ```

## How It Works


1. The app can read from  multiple PDF documents and extracts their text content.

2. The extracted text is divided into smaller chunks that can be processed effectively.

3. The application utilizes a language model to generate vector representations (embeddings) of the text chunks which is done locally here.

4. When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

   
