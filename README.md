# Network Intrusion Detection System (NIDS)

An intelligent, real-time Network Intrusion Detection System that leverages Machine Learning to identify and classify malicious network traffic.

## Project Overview
This project provides a secure, decoupled NIDS architecture. It utilizes a Python Flask backend to process incoming network metrics and IBM watsonx.ai to classify traffic as either 'Normal' or an 'Anomaly'. This project was developed using an AI-first approach with IBM Bob as a development partner.

## Features
*   **Real-time Analysis**: Processes network packets and provides instant classification.
*   **Machine Learning Powered**: Uses IBM watsonx.ai for intelligent threat detection.
*   **Decoupled Architecture**: Separation of backend processing logic and user interface for better scalability.
*   **Secure**: Environment-based credential management for sensitive API configurations.

## Prerequisites
Before running the project, ensure you have the following installed/configured:
*   **Python 3.x**
*   **IBM Bob**: AI-powered development partner used to plan, code, and document this project.
*   **VS Code**: The integrated development environment (IDE) used to host the IBM Bob agent and manage the codebase.
*   **IBM Cloud account**: With access to Watson Machine Learning for model inference.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd NIDS_2.0
    ```

2.  **Create and activate a virtual environment**:
    *   **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up credentials**:
    *   Copy the example environment file:
        ```bash
        cp env.example .env
        ```
    *   Open the `.env` file and fill in your IBM WML `API_KEY`, `SPACE_ID`, and `URL` values.

## Running the Project

1.  **Start the backend server**:
    ```bash
    python app.py
    ```

2.  **Test the API**:
    In a new terminal, use the following `curl` command to send a test JSON payload:
    ```bash
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"duration":0,"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":232,"dst_bytes":8153,"count":5,"srv_count":5,"serror_rate":0.0,"rerror_rate":0.0,"same_srv_rate":1.0,"diff_srv_rate":0.0}'
    ```

## Technology Stack
*   **Backend**: Flask (Python)
*   **AI Engine**: IBM watsonx.ai / Watson Machine Learning
*   **Development Partner**: IBM Bob (AI-driven SDLC orchestration)

---
*Developed as an automated network security solution.*
