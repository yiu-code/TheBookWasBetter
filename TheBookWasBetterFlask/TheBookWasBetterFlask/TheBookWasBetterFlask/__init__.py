"""
The flask application package.
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import yaml
import json
app = Flask(__name__)
db_config = yaml.load(open('./TheBookWasBetterFlask/database.yaml'))
app.config['SQLALCHEMY_DATABASE_URI'] = db_config['uri']
db = SQLAlchemy(app)
CORS(app)


import TheBookWasBetterFlask.views
