# AWS Chatbot with Langchain and Nemo Guardrails

## 📜 Description
> The application showcases the integration Langchain with documents loaded and Nemo Guardrails. By combining these technologies, the application ensures advanced safety features and effective mitigation's, enhancing the overall security and reliability of the chatbot system.

## 🛠️ Installation

#### Clone the repo
 ```
 git clone https://github.com/SSK-14/chatbot-gradio.git
 ```

#### If running for the first time,

1. Create virtual environment

    ```
    $ pip3 install env
    $ python3 -m venv env
    $ source env/bin/activate
    ```

2. Install required libraries

    ```
    $ pip3 install -r requirements.txt
    ```

#### Activate your virtual environment

```
$ source env/bin/activate
```

#### Create an .env file in the root with

```
OPENAI_API_KEY = "Your openai API key"
```

#### Run the Gradio app

```
gradio app.py
```

## 📁 Project Structure

```
Chatbot/
│
├── config  // Contains all files for Guardrails 
├── knowledge_base // Documents need for the chatbot context
├── vectorstore // Contains vector embedding pickle file
├── app.py // Main file to run
├── create_index.py // Run this to update vectorstore
├── README.md
└── requirements.txt

```

## Loading the Vectorstore 🗃️ 

1. Keep you data or documentations in the knowledge_base folder
2. Run the command - `python create_index.py`

## Contributing 🤝
Contributions to this project are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the project's GitHub repository.

## License 📝
This project is licensed under the [MIT License](https://github.com/SSK-14/chatbot-gradio/blob/main/LICENSE). Feel free to use, modify, and distribute the code as per the terms of the license.
