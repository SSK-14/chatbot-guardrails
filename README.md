# 💂🏼 Build your Documentation AI with Nemo Guardrails

![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2.svg?style=for-the-badge&logo=Google-Gemini&logoColor=white)
![Nvidia Nemo](https://img.shields.io/badge/NVIDIA-76B900.svg?style=for-the-badge&logo=NVIDIA&logoColor=white)

## 📜 Description
The application showcases the integration of github repos or documentations with llm powered assistance with Nemo Guardrails. By combining these technologies, the application ensures advanced safety features and effective mitigation's, enhancing the overall security and reliability of the chatbot system.

## 🚀 Demo

[NeMo Guardrails Chatbot](https://ssk-14-nemo-ai.hf.space/)

| Without Guardrails |
|------------|
| ![Without Guardrails](./images/without-guardrails.png) |

| With Guardrails |
|------------|
| ![With Guardrails](./images/with-guardrails.png) |

---

## 🛠️ Run The Application

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

#### Create an .env file from .env.example

- Get an [Gemini API key](https://makersuite.google.com/app/apikey) or [OpenAI API key](https://platform.openai.com/account/api-keys) or [Groq API key](https://console.groq.com) or Use local models using [Ollama](https://ollama.ai/).

- Setup a vector database [Qdrant Cloud](https://cloud.qdrant.io/).

Make sure you replace your key rightly.
```
# You can use your preferred models.
MODEL_API_KEY = "Your OpenAI/Gemini/Groq API Key"
QDRANT_URL = "Your Qdrant cloud cluster URL"
#If you are using qdrant cloud
QDRANT_API_KEY = "Your Qdrant API Key"
```

#### Loading the Vectorstore 🗃️ 

1. Update the constants in `vectorstore.py` and `nemo/config.py` <!-- Update env if with qdrant url and key. -->
- Change GITHUB_URL and BRANCH for your preferred github repo.
2. Run the command - `python vectorstore.py` <!-- Will create a vector collection. -->

#### Run the Gradio app

```
gradio app.py
```

#### Run using NeMo Guardrails UI

- Update the `nemo/config.yml` file with models and `export OPENAI_API_KEY=sk...` <!-- Update based on model provider. -->
```
nemoguardrails server --config nemo
```

## 📁 Project Structure

```
chatbot-guardrails/
│
├── nemo/  // Contains all files for Guardrails
├── app.py // Main file to run for gradio UI
├── vectorstore.py // Run this to create vectorstore
├── README.md
└── requirements.txt
```

## Contributing 🤝
Contributions to this project are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the project's GitHub repository.

## License 📝
This project is licensed under the [MIT License](https://github.com/SSK-14/chatbot-guardrails/blob/main/LICENSE). Feel free to use, modify, and distribute the code as per the terms of the license.
