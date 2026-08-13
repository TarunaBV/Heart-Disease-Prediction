import streamlit as st
import pickle
import numpy as np

with open('knn_model.pkl', 'rb') as file:
    knn = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

