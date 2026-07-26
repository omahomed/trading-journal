release: python migrations/run.py
web: sh -c 'uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}'
