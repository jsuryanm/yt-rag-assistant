import os 
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]: %(message)s")

project_name = "src"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/agents/__init__.py",
    f"{project_name}/agents/orchestrator.py",
    f"{project_name}/agents/transcriptor_agent.py",
    f"{project_name}/agents/summary_agent.py",
    f"{project_name}/agents/rag_agent.py",
    f"{project_name}/agents/mcp_tool_agent.py",
    f"{project_name}/core/__init__.py",
    f"{project_name}/core/config.py",
    f"{project_name}/core/state.py",
    f"{project_name}/core/llm.py",
    f"{project_name}/core/embeddings.py",
    f"{project_name}/mcp_tools/__init__.py",
    f"{project_name}/mcp_tools/mcp_client.py",
    f"{project_name}/a2a/__init__.py",
    f"{project_name}/a2a/agent_card.py",
    f"{project_name}/a2a/task_manager.py",
    f"{project_name}/a2a/a2a_server.py",
    f"{project_name}/tests/__init__.py",
    f"{project_name}/tests/test_transcriptor_agent.py",
    f"{project_name}/tests/test_summary_agent.py",
    f"{project_name}/tests/test_orchestor_agent.py",
    f"{project_name}/tests/test_rag_agent.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/custom_logger.py",
    f"{project_name}/exceptions/__init__.py",
    f"{project_name}/exceptions/custom_exception.py",
    "backend/api/__init__.py",
    "backend/api/main.py",
    "backend/api/routers/__init__.py",
    "backend/api/routers/summarize.py",
    "backend/api/routers/qa.py",
    "frontend/__init__.py",
    "frontend/streamlit_app.py",
    ".env",
    "requirements.txt",
    "Dockerfile.api",
    "Dockerfile.streamlit",
    "docker-compose.yml"
]

for file_path in list_of_files:
    file_path =  Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating an empty file: {file_path}")
    
    else:
        logging.info(f"{file_name} already exists")


