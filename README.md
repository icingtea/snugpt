# SNU-GPT

# How to run locally
1. Clone repo
```bash
git clone https://github.com/icingtea/snugpt
cd snugpt
```

2. Install requirements
```bash
uv sync
# or
pip install -r requirements.txt
```

3. Populate `.env`
```txt
MONGODB_CONNECTION_STRING=""
DATABASE_NAME="snugpt"
EMBEDDING_MODEL="intfloat/e5-large"
EMAIL_ADDRESS=""
EMAIL_PASSWORD=""
```

4. Run Seed scripts
```bash
uv run -m src.seeder.<each file>
```

5. Run App
```bash
streamlit run src/main.py
```