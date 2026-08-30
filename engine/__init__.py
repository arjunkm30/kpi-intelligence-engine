"""
engine/ — the pipeline, one file per stage.

Each module is independently labeled LLM or NOT-LLM in its own docstring,
so at any point you can answer "is this step a model call, or math/rules?"

  anomaly.py      -> NOT LLM  (detect_anomaly)
  attribution.py  -> NOT LLM  (attribute_change)
  evidence.py     -> NOT LLM  (retrieve_evidence — TF-IDF, no model call)
  confidence.py   -> NOT LLM  (rank_causes — weighted scoring + rules)
  narrator.py     -> LLM      (generate_narrative — the ONLY model-calling step)
  actions.py      -> NOT LLM  (recommend_action — lookup table)
"""
import os
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONTRACT_PATH = os.path.join(_ROOT, "kpi_contract.yaml")

with open(_CONTRACT_PATH) as f:
    CONTRACT = yaml.safe_load(f)

DATA_DIR = os.path.join(_ROOT, "data")
