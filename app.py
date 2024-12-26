from flask import Flask, render_template, request

import numpy as np

from sklearn.preprocessing import StandardScaler

from src.pip