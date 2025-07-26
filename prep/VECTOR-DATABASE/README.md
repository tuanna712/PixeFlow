# Welcome to this repo
## Python version and library required
-   Install uv: pip install uv
-   Create new enviroment: uv venv .venv
-   Activate new enviroment.
-   Install all library: uv sync

## Run this repo
To use this project: 
-   Create a Qdrant Cloud at https://cloud.qdrant.io
-   Get URL and API_KEY of your VectorDB on this website
-   Config server by: paste this URL and API_KEY in file config.yml
-   Run in terminal this command: uv run main.py

## Note for URL and API_KEY
Setup your URL and API_KEY of Qdrant in file .env with format:
-   QDRANT_URL=<URL>
-   QDRANT_API_KEY=<API_KEY>

Then, get URL and API_KEY by using:

```python
load_dotenv()
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")