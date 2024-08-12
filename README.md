# talk_with_arxiv_papers

Welcome to the talk_with_arxiv_papers Flask RAG Application! This project is a Retrieval-Augmented Generation (RAG) web application that allows users to query arXiv papers, view selected papers in PDF format, and ask questions related to the paper’s content.

## Features

- **Search Papers**: Query arXiv papers based on topics and receive relevant search results.
- **View and Interact**: Open selected papers to view the PDF and interact with the content via a dialogue box.
- **Ask Questions**: Engage with the paper by asking questions directly related to its content.

## Tech Stack

- **Flask**: Web framework for building the application.
- **Docker**: Containerization of the application for consistent development and deployment.
- **Amazon Bedrock**: For generating embeddings to enhance search and retrieval.
- **Meta LLaMA 3 Model**: Used for advanced language understanding and generation.
- **LangChain**: Manages text splitting and prompt engineering.
- **FAISS**: Vector storage and indexing for fast and efficient retrieval.

## Deployment

- **Heroku**: Platform used for deploying the application.
- **GitHub Actions**: CI/CD pipeline for automated deployment and updates.

## Getting Started

To get started with this project, follow these steps:

1. **Create a repository under your github account**

2. **Create following secrets in your repository**

    AWS_ACCESS_KEY_ID

    AWS_SECRET_ACCESS_KEY

    HEROKU_API_KEY

    HEROKU_APP_NAME

3. **Create a heroku account and a heroku app**

3. **Add following secrets in Heroku app console under settings and config vars**

    AWS_ACCESS_KEY_ID

    AWS_SECRET_ACCESS_KEY

    AWS_DEFAULT_REGION

4. **Copy this the repository in your repository**

5. **Push the code to your repository**

The deployment workflow is triggered on a push to the main branch.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

Contact
For any questions or feedback, feel free to reach out to me at rachit1405@gmail.com .