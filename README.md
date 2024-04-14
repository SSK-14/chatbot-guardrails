# AWS Chatbot with Langchain and Nemo Guardrails

![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2.svg?style=for-the-badge&logo=Google-Gemini&logoColor=white)
![Nvidia Nemo](https://img.shields.io/badge/NVIDIA-76B900.svg?style=for-the-badge&logo=NVIDIA&logoColor=white)

## 📜 Description
> The application showcases the integration Langchain with documents loaded and Nemo Guardrails. By combining these technologies, the application ensures advanced safety features and effective mitigation's, enhancing the overall security and reliability of the chatbot system.

## 🚀 Demo

[AWS Guard Chatbot](https://ssk-14-aws-guard-bot.hf.space/)

```
Note: It has only minimal guards added from NeMo for demo,
So its still prone to prompt injection 
```

| Without Guardrails |
|------------|
| ![Without Guardrails](https://github.com/SSK-14/chatbot-guardrails/assets/45158568/19945cb6-9346-451d-90ef-0cebe34400c9) |

| With Guardrails |
|------------|
| ![With Guardrails](https://github.com/SSK-14/chatbot-guardrails/assets/45158568/0cdc99cf-43b3-4028-958a-605511b35664) |

---

## 🛠️ Installation

#### Clone the repo
```
git clone https://github.com/SSK-14/chatbot-guardrails.git
```

#### If running for the first time,

1. Create virtual environment

```
pip3 install env
python3 -m venv env
source env/bin/activate
```

2. Install required libraries

```
pip3 install -r requirements.txt
```

#### Activate your virtual environment

```
source env/bin/activate
```

#### Create an .env file in the root with

```
OPENAI_API_KEY = "Your openai API key"
or
GOOGLE_API_KEY = "Your Gemini API key"
```

#### Loading the Vectorstore 🗃️ 

1. Keep you data or documentations in the knowledge_base folder
2. Get an [Gemini API key](https://makersuite.google.com/app/apikey) or [OpenAI API key](https://platform.openai.com/account/api-keys)
3. Update the embedding function in `create_index.py`
4. Run the command - `python create_index.py` <!-- Will create a vector database. -->

#### Run the Gradio app

```
gradio app.py
```

## 📁 Project Structure

```
chatbot-guardrails/
│
├── config  // Contains all files for Guardrails 
├── knowledge_base // Documents need for the chatbot context
├── app.py // Main file to run
├── create_index.py // Run this to create vectorstore
├── README.md
└── requirements.txt

```

## Contributing 🤝
Contributions to this project are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the project's GitHub repository.

## License 📝
This project is licensed under the [MIT License](https://github.com/SSK-14/chatbot-guardrails/blob/main/LICENSE). Feel free to use, modify, and distribute the code as per the terms of the license.
