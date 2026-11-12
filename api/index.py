import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obat_match import make_app, load_catalog

load_catalog()
app = make_app()