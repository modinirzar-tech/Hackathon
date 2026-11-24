export PYTHONPATH := src
.PHONY: install test run ui kpi graph pin redteam demo
install: ; pip install -r requirements.txt
test: ; python3 -m pytest -q
run: ; uvicorn assistant.api.main:app --port 8000
ui: ; streamlit run app.py
kpi: ; python3 scripts/kpi_report.py
graph: ; python3 scripts/build_graph.py
pin: ; python3 scripts/pin_manifests.py
demo: ; python3 scripts/demo.py
