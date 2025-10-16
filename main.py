import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


EMAIL_ADDRESS = "sender mail"
EMAIL_PASSWORD = "password" 
RECIPIENT_EMAIL = "receiver mail"
PRICE_CHANGE_THRESHOLD_PERCENT = 0.05 # 5% price change triggers alert

# --- Configuration for Styling (Improved & Unique Dark Theme) ---
st.set_page_config(layout="wide", page_title="Market Insights & Strategy Dashboard 📊")

# Custom CSS for a fresh look (less identifiable)
st.markdown("""
<style>
    /* Main body background color - a softer dark mode */
    .stApp {
        background-color: #1a1a2e; /* Deep purple-blue */
        color: #e0e0e0; /* Light grey text */
    }
    /* Sidebar styling */
    .css-1d391kg, .css-1dp5dkx { /* Sidebar container */
        background-color: #0f0f1d; /* Even darker blue for sidebar */
        color: #e0e0e0;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff; /* White headers */
    }
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #2c2c47;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4a4a6e;
    }
    [data-testid="stMetricLabel"] {
        color: #b0b0b0;
    }
    [data-testid="stMetricValue"] {
        color: #e94560; /* Reddish accent for values */
        font-size: 1.8em;
    }
    /* Info box */
    .stAlert {
        background-color: #2c2c47 !important;
        color: #e0e0e0 !important;
        border-left: 5px solid #e94560 !important;
    }
    /* Table styling for better dark mode visibility */
    .stDataFrame, .ag-theme-streamlit {
        background-color: #2c2c47 !important;
        color: #e0e0e0 !important;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- 1. LLM/NLP SETUP ---
@st.cache_resource
def load_sentiment_model():
    """Loads and caches the Hugging Face sentiment analysis pipeline."""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# --- 2. ENHANCED SIMULATED DATA GENERATION ---
@st.cache_data
def load_and_simulate_data():
    """Generates simulated data for pricing, competitors, and reviews."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    products_config = {
        'Zenith SoundPods Pro': {'category': 'Audio', 'base_price': 250, 'inventory_factor': 80, 'competitors': ['AcousticFlow Earbuds', 'SoundWave Buds'], 'image_url': 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEhUSEhIVFRUVFhcVFRUVFxUVFRUXFRUWFxUVFRUaHSggGB0lHRUWIjEhJSkrLi4uFx8zODMtNygtLisBCgoKDQ0NDg0NDjcZFRktKystKystKysrKysrKysrKysrKystKysrKysrKysrKysrKysrKysrKysrKysrKysrK//AABEIAOEA4QMBIgACEQEDEQH/xAAcAAEAAgMBAQEAAAAAAAAAAAAAAQYCBAUDBwj/xABKEAACAQMCBAIHAQsKAwkAAAABAgMABBESIQUGEzFBUQcUIjJhcZGBIzNCUlNiobHB0dIVFyRygpKisrPwJUNUFjRzdJPCw9Ph/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAH/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwD4nSoqaBUVNKBSujHwC7YBltbhlYAgiGUgg7gg6dxite94dNDjrQyRas6eojJnGM41AZxkfWg1qVFKCaUpQKVFTQKUr1mtnTSXRlDrqQspXWp7MpI9ofEUHlmlKyjQsQqgkkgAAZJJ7AAdzQY0FdP/ALO3v/R3P/oy/wANaFxA8bFJFZGHdXBVhkZ3U7ig86UpQKV6rbSFDIEYoCFLhToBPZS2MA/CvKgUqKmgUpSgippSgUqMVNANKUoFKVFBfuRL/il3KIV4hcQ2sCa55eoypBBGPPsDgYA+GewNa/OPH5eM30MEGoxKy29oshZmIYqplmc5Ys2AxJ7ADvgk2fifCrYWEfD7PivD44mxLdSSTES3E2cgMqqQsa4GBk9hncZapwWycJu7S7W7tbsJLqZLZy7Kq41ZyBglWOPiKDfTlHhklweHRX03rgZoxK8SCzeZTgxD2uoMkFQ248s7A6vL/JsElpc3V5cPbi0nEMqqgkY+BRR4uXIXvgbk7CrrPzBN13uP+0MIsSzSKkeg3oQ5KRC3aLIYHC5Pln4VToeMxPwfiCPKOtNexyqjsvVcEgs+kYz45IGKDS5o5btks4eIWM0rwSSNAyXCqs0cqqWGSnssCAe3bbc527fHeTOFWNx6tdX9wHkCMhjiRlhVlHt3BJ3y2rZBnABPeuPc8QiPAY7fqL1hxBpDHkawhgKhyvfGds1dPSXyva3PETI3Eba3wkIuUnYpKAI1w8C4xLlMbAjcfQKfD6P5BfXNrPMscVohmnuArMBCFDKyIPeZgwwufxu+N0/LNjc288/DbidntV6k0Fykau8QwGmiZDjC+IO/y2BsDc6Wt1f8Qjkcw2t9AttFMVJ6RiULFJIvfQx1E+IyM43I51lBb8Jtrxmvba5uLq2e0hjtJDKqpKR1ZJX0gLgAYHc/qDQtuV7KC1guOJXE0bXQLQQ28aOyx5wJpSxxg5BCjfHmcgWjm/lkXd3wawinUq9iqrPpOGRA79QITnJVMhc9yBmuRdRW/FbOyIvba1uLSEW00d05jVo0P3OSJsHWcZyvmfgM7fMd7w57vhSJeyC3gtViNzAdE0ToXEchGCUywUkbEBqDgcU4Hw0wzta3U8c9vjVBepHE0wzg9HS2dQ76SM1xeUP+/wBp/wCag/1kr6NxTipFndJxTiVjfqYWWzWEpLcLcH73LrVAYwN86j8PMH5ryxMqXlq7kKq3ELMx2CqsqksT4AAE0H1fm7hXMTXtw1tPKsBlYxAXkSAL4AIZAV+WBXzyHhb3dtxG+uJpGntWtgdRD9QzSmJtbnf2QgxjyrX5/uY5eJXckbK6PM7K6kFWB7EHxrschSQSWfEbGW5it3ultmhknOiEm3lZ2Vn/AAc5H6fLFByYuX0PC5OIa21pdLb9PA0kGMPqJ753xVlv+SeHW0y2N1eXEV0yBvWGijWxDMmsDUW1sozp1ds+XYRxdbW14K9mt7BcTm9SWRYGyoHT04jYgdQABcsBgFseFWPhXEem0ejjVrLwrCF7e+ZJbnpaR1ITCY9RI3ChTjt3GxCk23D5Bwa7k9YbRHeRxtCmlopGwv3TXjJ+GDgjFbM/K3D7MRR8SurhLmZFkKW8aOlsr+51ixyxxuQgyPjsTnccUtTwniEULqnU4gJIIScSdDbRhe+AuB9lbvMdna8YlS+TiFrbM8Ua3UNy5jkikjUKxiXH3VcAYxjPwycBo8M9HLNxN+HSzAAQtNFOgGiRNIMb75wDnfvgg7nvXmOVuH3FvctYXc8k1nEZ360SxxTxp98aEAlkx3AbfsPHIsfDObbSTjDSrKI7aGwa0hklOjqBEwrHPYsS2Ad8Y7Haql6PL+KJOJCWRUMnDp4owxA1yNp0ovmx8qCo1FTSgUpmlApUf78KmgUrDXTVQZ0rDXTVQZVNYa6a6DOlYa6a6DI11+aOPy39w1zMqK7KikRhguEUKMBmJ7Dzrja6a6DOorHVTXQZ0rHVTVQZUrHVTVQZUrHVTXQZUrHVTXQZUrDVTXQZ1FY66a6DOlYa6a6DOlYa6a6DPP8AvalYaqUGNKUoFKUoFKUoFKVNBFTipArILQYYqcV6rHWYhoNfFTitkQVPQoNTFbXC+HSXEqQwrqdzhR27DJJPgAAST4AGvSKzZ2CopZmOFVQWZj5ADc19G5W4M1hbSSSALcTEppypaOFcbHBOCzbn4IvmaBYcCs7Bd0S6n/CkkXVEp8oom2I/OcEnuAvapuOc7pdkldR4BSVUf2RtXMu7gsTmuNdSUF34LxGx4k4teJwRh5PZiu4lWGUOfdWRlGG+BIIz3BzmuR6QfRO3DoWuY7lZY1I9hl0ShWYLnAJD4LLkjHfNVRXOdv8AflXa5u5iuLiOMyOWzHp3PhjGPrmgoxqKyxWNApSlApSlApSlApSlApSlApSlApSpoAqQKAV6olAjjJIAGSTgAbkk9gB4mrjZ+j650LJdNHZo3u9ct1XHmsCgt5e9p719K9G8nDuHwZYR+shC0s7YLg4y0aEjKqMYwO5GfGvmPHOPy3U7zysSznIGfdX8FB8AP3+NB005GgPucRQn8+3dV/vB2P8AhrU4lydcQLrZVkj7dWJupGD5Me6f2gK04LsjxqwcD5lkhYEN8CO4YHuGB2IPkaCtrw8+VZfycfKvrtqeFPEspspGLEhxC50q3cbGQYB3wB5EVI/kr/oLr+8P/toKrY2qcPiwAOu6/dH8VB36anwA8fM/IVWOM8ScnUrEEHNTxPiD9WaKTOUYlNXvGMn2QfiMj6iuFPdZoOpLxaN1DAaW7OvkfMfA1y558mtG5TIJHcb/AGeNeEUpoO3Zx53NbkfDerFL5RRu4+SAucfYDVfkumXbcGrXymS0Fx8beYfWJ6CkOteZFdG5gxWk60HlSpqKBSlKBSlKBSlKBSlKBSlKBWQFRWaCgzjSrhylye10pnmkW3tU9+d8YOPeWMEjUfDPYHzO1aHJ3BBdXARzpiRTLO/bTEm7b+BOQM+Gc+FWrivGoyq3cyD1aP2bCz7IQowkjr4kjcZyAPMmg4PMVlNDM0co0k5OO/ffFcJzvV19Jdxqm1nvhT/hFUi5lB389/rQbUD1vQISQB3NcJZ8Vt8O4m4cFO4BIJ3A+OPGgvPCbxIybaUalYYlXJGOxUBlIKsNjkfDzNVznLgctqyyQzyvbyHCEudSNjPTfHc4BIPjg+Vato+DnJJ7knuSTkknzzXavuMD1WSNt9akKD+MBqVh8iufsoKxw6/QErcaiQG0PuSCykaW8xnf4H9GrMPI5Hwrt2Nta3gEbOIJ+ylvvch8ifA1zuN8sXNqcSxnHgw3U/EGg1I5wvvb/Dz+Hwro8J5iW3V1jtYizqyGST7o6hlKkoWHsHBO64rgEUXuPnQWDj9qFEZHii/qq0+jB2DOUVWYJIVV/cY9NsB9jsT32ri8zp9zi/qL+qtnlBsQXW+P6Ncf6ElBYb/gdtfMyRRepXoBb1dyOhPjv0W7L9mB5qvevm/ELR43ZHUq6EqysMMpGxBFWrljjvrai0uWPUHtW0+fuiOu4Grz/X2NdLnK1N3a+usoW5t3FteAdm7CKYfVR8nUfg0HzRhWJr2lWvI0GNKUoFKUoFKUoFKUoFKUoJAr3iWvJRW1brQXbl2HTwy6YbNPPBbE+SFlLAfMSEVwOfLvVc9MbJEqqo8BkAn9g+yrVytCZbC6gUZdGjuox4sYmBYD7FUfbVR5ygzMJl3SVQwPhkDcfqoLF6Rj7YP5qf5FqveoBrfq+KD613/SF3X+on+Ra0OHrmzYee312oKqSD2Nbtv7CZ8W3+zwH7ftqw2fJQnTVBMj7bjsfpWhdctTQ+ywP7PsoOdHc71v/wAl3FwdEETysF91FLaQ25Zj+CDgKM/GtSSzVUY76gMjevvfJfDktbKFFxqZFklbxeR1BJJ8QMhR8FFB8F4lypf26GSa0mRB3codI+bDYfbVr5G5ubR6rcYkj7Lr30/DJ8K+3i4BUq2GBBBU7gg7EEHuDX5749w2O04jcRRZ6aMCgO+A6K4XPjjVj7KDPmqxtQS0Y0nyG4qnAb/bVijspLhsDxq1cP5EjVdUrUHF5nYGOLB/AX9VY8sSAQ3GSBmCYb/nROAPqcVp8WtbZJWSS5kKDOkxxh2+CnLgZ8M/qrjcQeHOIDIU/GlCqx/sqSB9TQa0ExRldTgqQwPkQcivs3C06z3kfZbnh5cjydA4Vvnuv90V8g4XYtPIsajudz5DxNfW+EyaIb677KsHqkJ82OQ2PPDOo/snyoPks61qtXSu0rnuKDzpSlApSlApSlApSlAqRUVIoM0rp8LgLuqDuzBR8ycD9dcxKuvottw/ErUMuoBy+CcDMaNIuTg+Kg/ZQfVOB+js2cqlrsasdhGQCDsQTqNZ8R9F1tLrHrDKjsW6ekFUJ76D3G+/wzU8084hJcmNtttiD+vFVuT0oINulJ/g/ioOhzP6MnucaLuMYAHtRt4ADwPwrwsfRZIkRiN3H3zkRvjb4ZrSPpWT8lL9I/46D0qr+Tk+ifx0E8O9EM8DhlvxpHfRGytnwxkkVaU5QlK6ZZxKPNkYH64qsD0rL+Tk+ifxV6r6WE/Jv9F/ioMr70UGQn+kaQcjARjjNW6LhE6oqBo8KoX/AJg90Afi/Cql/Own5J/ov8VT/OxH+Rf9H76C4R8Mn8Sn2az+wVVONejBrm5kuDcFTJpyghLY0oqe9rH4vlXl/Osn5NvoP30HpVT8R/ov76Dp8F9Hpg7SsSfHokf+6o416PLi42F+Y1/FEAJ+vWFaH866fiP9B++sG9KifiP9B++g5svoPbuL7PzgA/8Amryj9CTA+1d5HkI1B+vUNdb+dFT+C30//agekgH8E/SgQ+jXooUjuNGcamCKWx44OvY/HfHlVg5g5Id7NY7Vl6UKlhHg6mKqds5OT379ya5Fvz3EfeWT7AD+tqvXKXMccyhVV/mwUD/MaD8ycRTBrkyVcvSHbpHfXKRghBK2AfDO5HyBJA+GKpslB5GlSaigUpSgUpSgUpSgUFKCg9Eq++h4/wDE4P6s3+hJVBWrv6I5dPFbb87qr9YJcfpoLVzsvtt8zXzu6O9fTeeYvab7a+ZXnc1BqmozUE1GaKzzWQasFUnsCfkCaMCO4I+YIoPTVU6q8tVA1VHsDWQrxDVkHoPWpFYxKzHCqWO5woLHABJOB5AEnyANQGoPZa24TWgHraheg6tvX1P0cdxXyq0NfWfRynY0HyL0lN/xG7/8Z/11TZDVk55uNd9dN5zy/okYfsqstQYGlKUClKUClKUClKUClKUGQrs8r8S9WuoJ/CKWNz/VDDX/AIc1xRXrGaD736QLb2iRuDuD5g9jXyHiK4Y19J4PxcXvDo8nMsI6UnmdIADfauk/aR4V845h9ljUHHkkrz6teTNUZoqwcszXLSGOC5kt1ILzOkjxqkae9I4UjOAdh4kgeNdh5zxSd2Z5ehbRAIGZWmZNYVcvIca3ZtTMTgYxvgCq1wrjj26SRrHE6ylS4kTXnRuo7jbO+POvQcwuH1pDAgKlHRYgI5FYgkSLnfcAjsRiqjuycuwrJ7UsixeryzkZieZDCRlG0HQ2Qcg7d98YNeMHB7eRoXWWVIZYrhyXVGkja2Vi3u4Dg4HbB7j41w344+piqRIGiaHTGgVQj+98SfiSa2uBcdMZQMwUQxXQiOnV7c0TaQRvnLae4x50G9b8LiuQptHkz1ooXWcJletq0SBk2I9hsr37bmsGtLaVJTayTM8IDkSqirKmtULJp3Qgupw2dvHNcy54/KwRUWOEI4lAhXRmRfdc7kkjw8BntU33MEkqOmiKPqEGUxRhGlIORrPlnfAwM0Fo4TbWsN20IllaeGO5ViVQQu4tpVkRB7y4ycMc509hkVy7zhcaWizoZZDpjZnXpNArPjVG4DdSMrnGW7kdhkGtMc1T5LaIeoylHl6Y6kishQh2z3we4wTgZzvWrJxx+kYhHCmpFjd0jCyOikEKzDbuqk4AJxvQeIuK3La4Brjaq9EkqKt/D3yRX1/lW4W3tJJ22EcbOfkqk/sr4VwniYVhq7fCrBzJzo0sAtYQUj/DPi58AfIfCqinX05dizHJYlj8ycmtJq9ZGrxNBFKUoFKUoFKUoFKUoFKUoFZqawqRQd3lzjLW0moe6w0uPMeBx5jJ+wkeNbPMzrIOohGM7/b+yq6rVsRynGP0HtQahqK9pE8vpXiagUpSqFKUoFKUoFKUoFSKis0TNB6RVlI9QTjYV5MaCGNYmhNRQKUpQKUpQKUpQKVlppoNBjSstNNNBjSstNRpoANZq1Y6aYoPcP50MQPjivIVkGxQDbt5Z+VYNGR3B+le6zYr1W6oNGldAXg8qy9bX8X9AoObSun66vl+gVHro8v1UHPWNj2B+legtm+XzrYe7+deTTUARAd96h3rAtWNAJrA1lio00GNKy000UGNKnTU6aDGlZaaaKDGlZaDSgzoaUoFRU0oFKV2+AcHS5gvG1MJreFbiNBjQ8auFuNW2cqrAjB8DQcSlX259HmPUFjd2kuJY4LtSB/R5ZY4p1UDG4EUhJ77oflXieToJ2tTZSSPHPeT2sjuUPTEbhkkBAHeAlz3900FIpV2v+ToIpbx+pKbSG0S6t5fZDS+sBBbK2V2y7MDsPvZ7Vv2Ho9gkvpYTO62vRgkt5zp1SNedNbVTtjdnYHA/wCWe1B86pXe4hwEQWUc8utZ5bmWJY9tPStwFlY7Z1dVtPf8E178l8LtLqRbeVLtppHwnQMKoEC5LSFwSMYYk9gBQVqlX2w5Mtbie7e2a7uLO1MaDoIslzcSSZGI8DSqAqzFyPdA2Oa95/R7H6xYgLdwwXcrQulyipcQuuD306WVgcghfAg0HzulXG45fsJIbr1Sa4aezTqO0ojEM6LIscjRKvtR7sCNROR5eHvf8C4VaLbesy3rvcWsFwywCACHqoCclx7e+cKMYA3JzQUelXY8jxwzXZuZ2NpaJDIZIVHUnW5x6uI1bZS2TknIXSe/ete45at7iFbjhrTEesR20sNz09aPN95kEiAKyMfZ7AgjxFBV7G0eaWOGManldY0XIGWdgqjJ2GSRuaxuIWjdkcYZGKsNjgqSCMjY7g19G4BwjhUHFLe1E90bmG7iUzFYvVnmilUtGI/vgBZSgbJ3IOMVha8oQTi4upVvJ83k8bJYrG7W4V8h5kYFm1aiQFGMKcnwoPnFe97ZSwvomjeN8AlHUo2GGQdJ33BFeU6gMwXJAJALDSxAJxqUE6T5jJx519Au+V4Xvb1Lm6uGS1tIbjqsVlmYGO3Zo/a77Ssq7gD2c7A0Hz2lWXj/AAe19VivbJp+k0z28kdxoMiSIiyAq0YAZSreQwR4+FaoFKYpQKUoKCKmo+lTQKVFTQKUNKBXW5X4z6ncrOY+qgDpJCTpWWOVGjdGODsQ3kewrk0oLlw/0gzRSX0pjDteF3Ql97aVlkRJY/Z3KpK642zgeVaHLXNjWdvc24i19dfubltJt5DHJC0qDByxjlZe4quUoLJd82vJw6Ph/TwUcap9RLSRRtK8ULLj3VaZyN/Lasr7nGSSxt7QJoeBlJuA3tyLE0rW6EYGkR9Z8bnwqs0oLTzPzct9dQzy2wWGLGu3WQgSFpWluDrABQyM7ZIGRtWfBuaLSCO7jFlIPWnYaorkJJHbEgi1DvE5K7e02xYAA9qqVKC08L5ltoevCLWQ2dwseuIzgzI8RJSWOYRgAjUfZK4OcGpg5ltoLq1ntrV1S2fW3VnZ5Zznu7adCYGwCr4nOfCrYpQdjhXHOgLsdPV61A0PvY6eqVJNXY6vcxjbvVw5tv7FRYLd2ckrJw6zZWin6PUUwjEUwKNtnPtLht8b4FfNqzeRjjJJwABkk4A7AeQHlQWxeeGknuXuYFlt7tUjlt1YxhFhx6uYXAJVkwMbYOWyN68L3miOOBLbh8TwRrMtw0kkglnllj+9FiFVVVcnCgd96rFKC8xc52S3S8Q/k5jddQSOvXxbGTUGkmSPQXDn2iAWIUkHfGDqcI5ms45fWJLObrrO86yQXJi1hn1LDKCpwo7akwTVQqRQdnifGIbgXLyWw9ZnuGnEyyOFjVyWeIRdm3PvHeulf85dSa9l6GPW7WO209TPT6awLrzp9rPQ7be932qqUoOs3Gf6ALLp9rprnqavxoVi0aMfm5znx7VyaUoFKUoFBQ1FApU0oFDSlAPajf7+tKUA1PnUUoJqBSlBNYippQKUpQRSlKCRUUpQZVj+791TSgCoFTSgUWlKAaClKCDU+NKUA0NKUClKUH//2Q=='},
        'Visionary HD Monitor': {'category': 'Displays', 'base_price': 400, 'inventory_factor': 60, 'competitors': ['ClarityScreen X1', 'ViewPro Display'], 'image_url': 'www.google.com'},
        'PowerCore Laptop': {'category': 'Computers', 'base_price': 1200, 'inventory_factor': 30, 'competitors': ['SwiftBook Air', 'MegaCompute Pro'], 'image_url': 'data:image/jpeg;base64,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'},
    }

    all_pricing_data = []
    
    for date in dates:
        for product_sku, config in products_config.items():
            base_price = config['base_price']
            
            # Simulate a subtle price change over time + noise
            trend_factor = np.sin((date - dates.min()).days / 30) * 5 # Small cyclical change
            current_price = base_price + trend_factor + np.random.uniform(-5, 5)
            
            # Simulate inventory
            inventory_factor = config['inventory_factor']
            sales_velocity = np.random.normal(10, 3)
            current_inventory = max(0, int(inventory_factor - sales_velocity * (date - dates.min()).days / 30))

            # Simulate our promo
            is_promo_self = (date.month % 3 == 0 and date.day > 15 and np.random.rand() < 0.7) or \
                            (current_inventory < 0.2 * inventory_factor and np.random.rand() < 0.5)

            final_price = current_price * (0.85 if is_promo_self else 1.0)
            
            competitor_prices = {}
            for comp in config['competitors']:
                comp_base_price = base_price * np.random.uniform(0.9, 1.1)
                comp_is_promo = (date.day % 7 == 0 and np.random.rand() < 0.3)
                comp_final_price = comp_base_price * (0.92 if comp_is_promo else 1.0)
                
                # Only AcousticFlow Earbuds gets the target variable for the ML model
                if comp == 'AcousticFlow Earbuds':
                    competitor_prices[f'Competitor_Promo_{comp.replace(" ", "_")}'] = int(comp_is_promo)
                
                competitor_prices[f'Competitor_Price_{comp.replace(" ", "_")}'] = comp_final_price

            all_pricing_data.append({
                'Date': date,
                'Product_SKU': product_sku,
                'Category': config['category'],
                'Our_Price': final_price,
                'Our_Inventory_Level': current_inventory,
                'Is_Our_Promo': int(is_promo_self),
                'Image_URL': config['image_url'],
                **competitor_prices
            })
            
    df_pricing = pd.DataFrame(all_pricing_data)
    df_pricing['DayOfWeek'] = df_pricing['Date'].dt.dayofweek
    df_pricing['Month'] = df_pricing['Date'].dt.month
    df_pricing['Quarter'] = df_pricing['Date'].dt.quarter

    # --- Simulated Customer Reviews Data ---
    all_reviews_data = []
    review_templates = {
        'Visionary HD Monitor': [
            {"text": "The 4K resolution is stunning, crisp and clear visuals.", "sentiment": "POSITIVE"},
            {"text": "Colors are a bit washed out, not as vibrant as expected.", "sentiment": "NEGATIVE"},
            {"text": "Easy to set up, plug and play. Good stand too.", "sentiment": "POSITIVE"},
            {"text": "Wobbly stand, feels cheap for such an expensive monitor.", "sentiment": "NEGATIVE"},
        ],
        'Zenith SoundPods Pro': [
            {"text": "Fantastic noise cancelling, really blocks out the world!", "sentiment": "POSITIVE"},
            {"text": "Battery life is terrible, constantly needs charging.", "sentiment": "NEGATIVE"},
            {"text": "Sound quality is decent, but nothing extraordinary.", "sentiment": "NEUTRAL"},
        ],
        'PowerCore Laptop': [
            {"text": "Incredible performance, handles all my heavy software with ease.", "sentiment": "POSITIVE"},
            {"text": "Gets extremely hot under load, the fan noise is unbearable.", "sentiment": "NEGATIVE"},
            {"text": "Battery life is surprisingly good for a gaming laptop.", "sentiment": "POSITIVE"},
        ]
    }

    for date_idx, date in enumerate(dates):
        for product_sku, templates in review_templates.items():
            if np.random.rand() < 0.2:
                template = np.random.choice(templates)
                all_reviews_data.append({
                    'Date': date,
                    'Product_SKU': product_sku,
                    'Review_Text': template['text'],
                    'Rating': np.random.randint(1, 6)
                })
    df_reviews = pd.DataFrame(all_reviews_data)
    
    return df_pricing, df_reviews

# --- 3. MACHINE LEARNING MODEL TRAINING (WITH FIX) ---
@st.cache_resource
def train_competitor_promo_model(df):
    """Trains a Random Forest model to predict a key competitor's promotional move."""
    
    target_competitor_promo_col = 'Competitor_Promo_AcousticFlow_Earbuds'
    
    if target_competitor_promo_col not in df.columns:
        st.warning(f"Target competitor promo column '{target_competitor_promo_col}' not found. Skipping predictive model.")
        return None, None, None, None
        
    # FIX: Drop rows where the target is NaN (only AcousticFlow competitor data has a non-NaN target)
    df_clean = df.dropna(subset=[target_competitor_promo_col]).copy()

    if df_clean.empty:
        st.warning(f"After cleaning, no data remains for the target column: {target_competitor_promo_col}.")
        return None, None, None, None

    features = ['Our_Price', 'Our_Inventory_Level', 'DayOfWeek', 'Month', 'Quarter', 'Is_Our_Promo']
    X_cols = [f for f in features if f in df_clean.columns]
    
    X = pd.get_dummies(df_clean[X_cols], columns=['DayOfWeek', 'Month', 'Quarter'], drop_first=True)
    y = df_clean[target_competitor_promo_col]
    
    model_feature_names = X.columns.tolist()

    if len(y.unique()) < 2:
        st.warning("Target variable has only one class after cleaning. Cannot train classification model.")
        return None, None, None, None
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, 
        stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    feature_importances = pd.Series(model.feature_importances_, index=model_feature_names).sort_values(ascending=False)
    
    return model, feature_importances, {'Accuracy': accuracy, 'Report': report}, model_feature_names

# --- 4. SENTIMENT ANALYSIS IMPLEMENTATION ---
def analyze_sentiment(df_reviews):
    """Applies the pre-loaded sentiment pipeline to the review text."""
    if df_reviews.empty:
        return df_reviews
        
    review_texts = df_reviews['Review_Text'].tolist()
    results = sentiment_pipeline(review_texts)
    
    sentiments = [(r['label'], r['score']) for r in results]
    df_reviews['Predicted_Sentiment'] = [s[0] for s in sentiments]
    df_reviews['Confidence'] = [s[1] for s in sentiments]
    
    return df_reviews

# ----------------------------------------------------------------------
# --- 5. EMAIL ALERT LOGIC ---
# ----------------------------------------------------------------------

def send_email_alert(sku, price_change_pct, old_price, new_price, image_url):
    """Sends an HTML email notification for a significant price change."""
    
    # Determine alert type and color
    change_type = "DROP" if price_change_pct < 0 else "INCREASE"
    # Green for price drops (often a sales trigger), Red for price increases (margin opportunity)
    change_color = "#4CAF50" if change_type == "DROP" else "#e94560" 

    # Calculate absolute change in dollars
    price_diff = abs(new_price - old_price)
    
    subject = f"🚨 PRICE ALERT: Significant {change_type} for {sku} ({abs(price_change_pct):.2%})"

    # HTML Email Content
    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h2 style="color: #1a1a2e;">Market Price Shift Detected!</h2>
                
                <div style="border-left: 5px solid {change_color}; padding: 10px; margin-bottom: 20px; background-color: #f9f9f9;">
                    <p style="font-size: 1.1em; color: {change_color}; font-weight: bold;">
                        The price for {sku} has experienced a significant {change_type}!
                    </p>
                </div>

                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                    <tr>
                        <td width="150" style="vertical-align: top; padding-right: 20px;">
                            <img src="{image_url}" alt="{sku} Image" width="150" style="display: block; border-radius: 4px;">
                        </td>
                        <td style="vertical-align: top;">
                            <h3 style="color: #333333; margin-top: 0;">{sku}</h3>
                            <p style="font-size: 1.2em; line-height: 1.5;">
                                <strong>Price Change:</strong> <span style="color: {change_color}; font-weight: bold;">{abs(price_change_pct):.2%} {change_type}</span> (${price_diff:.2f})<br>
                                <strong>Previous Price (5-day Avg):</strong> ${old_price:.2f}<br>
                                <strong>Current Price:</strong> ${new_price:.2f}<br>
                                <strong>Alert Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            </p>
                        </td>
                    </tr>
                </table>
                
                <p style="margin-top: 20px; color: #555;">
                    *Strategic Note:* Review the *Competitor & Predictive Insights* section of the dashboard to determine if this change was in response to competitor movement or a strategic inventory adjustment.
                </p>
                <p style="text-align: center; margin-top: 30px;">
                    <a href="http://localhost:8501" style="background-color: #1a1a2e; color: #ffffff; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">
                        Go to Dashboard
                    </a>
                </p>
            </div>
        </body>
    </html>
    """

    # Email Setup
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))

    # Send Email
    try:
        # Use an appropriate SMTP server (e.g., 'smtp.office365.com' for Outlook, or 'smtp.gmail.com' for Gmail)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls() 
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, text)
        server.quit()
        st.sidebar.success(f"Email alert successfully triggered for *{sku}*!")
    except Exception as e:
        # NOTE: This error is common if the App Password or SMTP server/port is wrong.
        st.sidebar.error(f"Failed to send email. Check SMTP settings and App Password. Error: {e}")

def check_and_alert_price_change(df_pricing_data):
    """Checks all products for a price change over the threshold and sends alerts."""
    
    alert_triggered = False
    
    for sku in df_pricing_data['Product_SKU'].unique():
        # Get the latest 7 days of data for the product
        df_sku = df_pricing_data[df_pricing_data['Product_SKU'] == sku].sort_values('Date').tail(7) 
        
        if len(df_sku) < 2:
            continue

        # Latest price is the 'new' price
        new_price = df_sku['Our_Price'].iloc[-1]
        image_url = df_sku['Image_URL'].iloc[-1]
        
        # 'Old' price is the average of the previous 5 days (to smooth out noise)
        if len(df_sku) >= 6:
            # Average prices from 6th-to-last day up to 2nd-to-last day
            old_price = df_sku['Our_Price'].iloc[-6:-1].mean()
        elif len(df_sku) == 2:
             # Fallback: Just compare to the single previous day
            old_price = df_sku['Our_Price'].iloc[-2]
        else:
             # Not enough data, skip
             continue 
            
        price_change_pct = (new_price - old_price) / old_price

        if abs(price_change_pct) >= PRICE_CHANGE_THRESHOLD_PERCENT:
            alert_triggered = True
            send_email_alert(sku, price_change_pct, old_price, new_price, image_url)

    if not alert_triggered:
        st.sidebar.info("No significant price changes detected that meet the alert threshold (5%).")

# ----------------------------------------------------------------------

# --- STREAMLIT APP EXECUTION ---
df_pricing, df_reviews_raw = load_and_simulate_data()
df_reviews_analyzed = analyze_sentiment(df_reviews_raw.copy())
model, importances, metrics, model_feature_names = train_competitor_promo_model(df_pricing) # Train model once

st.title("Market Insights & Strategic Adjustments Dashboard 📊")
st.markdown("---")

# Sidebar for Navigation and Filters
with st.sidebar:
    st.header("Navigation & Filters")
    page_selection = st.radio(
        "Go to:",
        ["Product Performance Analysis", "Competitor & Predictive Insights"]
    )
    st.markdown("---")
    
    selected_product_sku = st.selectbox(
        "Select Your Product SKU:",
        df_pricing['Product_SKU'].unique()
    )
    
    # NEW: Email Alert Trigger Section
    st.markdown("---")
    st.header("Proactive Alerts 📧")
    st.write(f"Threshold: *{PRICE_CHANGE_THRESHOLD_PERCENT:.1%}* change vs. previous 5-day average.")
    
    if st.button("Manual Price Alert Check", use_container_width=True):
        check_and_alert_price_change(df_pricing)

# Filter data based on selected product
df_our_product_pricing = df_pricing[df_pricing['Product_SKU'] == selected_product_sku].sort_values('Date')
df_our_product_reviews = df_reviews_analyzed[df_reviews_analyzed['Product_SKU'] == selected_product_sku]

# ----------------------------------------------------------------------
# --- PAGE 1: Product Performance Analysis (with Reviews Table) ---
# ----------------------------------------------------------------------
if page_selection == "Product Performance Analysis":
    st.header(f"Product Performance for: *{selected_product_sku}*")
    st.markdown("Understand customer satisfaction and product trends.")

    col_a, col_b = st.columns([0.6, 0.4])

    with col_a:
        st.subheader("Customer Sentiment Distribution")
        if not df_our_product_reviews.empty:
            sentiment_counts = df_our_product_reviews['Predicted_Sentiment'].value_counts(normalize=True).reset_index()
            sentiment_counts.columns = ['Sentiment', 'Proportion']
            
            sentiment_color_map = {'POSITIVE': '#4CAF50', 'NEGATIVE': '#d62728', 'NEUTRAL': '#ffb02e'}
            
            fig_pie_sentiment = px.pie(
                sentiment_counts,
                values='Proportion',
                names='Sentiment',
                title=f"Overall Sentiment for {selected_product_sku}",
                color='Sentiment',
                color_discrete_map=sentiment_color_map,
                hole=0.4
            )
            fig_pie_sentiment.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e0e0e0")
            st.plotly_chart(fig_pie_sentiment, use_container_width=True)
        else:
            st.info("No reviews available for this product yet.")

    with col_b:
        st.subheader("Key Performance Indicators")
        if not df_our_product_reviews.empty:
            positive_reviews = (df_our_product_reviews['Predicted_Sentiment'] == 'POSITIVE').sum()
            total_reviews = len(df_our_product_reviews)
            customer_sentiment_score = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0
            
            st.metric(label="Customer Sentiment Score", value=f"{customer_sentiment_score:.1f}%")
            st.metric(label="Total Reviews", value=f"{total_reviews}")
            st.metric(label="Average Rating", value=f"{df_our_product_reviews['Rating'].mean():.1f} / 5.0")
        else:
            st.metric(label="Customer Sentiment Score", value="N/A")
            st.metric(label="Total Reviews", value="0")
            st.metric(label="Average Rating", value="N/A")

    st.markdown("---")
    st.subheader("Sentiment Trend Over Time")
    if not df_our_product_reviews.empty:
        df_our_product_reviews['Month'] = df_our_product_reviews['Date'].dt.to_period('M').astype(str)
        sentiment_trend = df_our_product_reviews.groupby('Month')['Predicted_Sentiment'].apply(
            lambda x: (x == 'NEGATIVE').sum() / len(x) if len(x) > 0 else 0
        ).reset_index(name='Negative_Sentiment_Rate')
        
        fig_trend = px.line(
            sentiment_trend,
            x='Month',
            y='Negative_Sentiment_Rate',
            title="Negative Sentiment Rate Over Time",
            markers=True,
            line_shape='spline',
            color_discrete_sequence=['#e94560']
        )
        fig_trend.update_yaxes(tickformat=".1%", range=[0, 1.0])
        fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e0e0e0",
                                 xaxis_title="Month", yaxis_title="Negative Sentiment Proportion")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No sufficient review data to display sentiment trends.")

    st.markdown("---")
    st.subheader("Raw Review Data and LLM Output Table")
    if not df_our_product_reviews.empty:
        # Display the review table, similar to the reference image
        display_cols = ['Date', 'Review_Text', 'Rating', 'Predicted_Sentiment', 'Confidence']
        st.dataframe(
            df_our_product_reviews[display_cols].sort_values(by='Date', ascending=False),
            column_config={
                "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                "Review_Text": st.column_config.Column("Review Text", width="medium"),
                "Rating": st.column_config.ProgressColumn("Rating", format="%d/5", min_value=1, max_value=5),
                "Predicted_Sentiment": st.column_config.TextColumn("Sentiment"),
                "Confidence": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1)
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No raw review data to display.")

# ----------------------------------------------------------------------
# --- PAGE 2: Competitor & Predictive Insights (with Pricing Table) ---
# ----------------------------------------------------------------------
elif page_selection == "Competitor & Predictive Insights":
    st.header(f"Competitor Dynamics & Predictive Insights for: *{selected_product_sku}*")
    st.markdown("Anticipate competitor moves and adjust your strategy proactively.")

    if model is not None and importances is not None:
        col_c, col_d = st.columns([0.6, 0.4])

        with col_c:
            st.markdown("#### Model's Key Predictors for Competitor Promos")
            fig_importances = px.bar(
                importances.head(7),
                orientation='h',
                labels={'value': 'Feature Importance Score', 'index': 'Feature'},
                title="Top Factors Influencing Competitor Promotions",
                color_discrete_sequence=['#4CAF50']
            )
            fig_importances.update_layout(showlegend=False, yaxis={'autorange': "reversed"},
                                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e0e0e0")
            st.plotly_chart(fig_importances, use_container_width=True)
        
        with col_d:
            st.markdown("#### Model Evaluation Metrics")
            st.metric(label="Model Accuracy", value=f"{metrics['Accuracy']:.2f}")
            st.markdown("---")
            st.markdown("#### Promo Class Metrics (Key Competitor)")
            promo_class_label = '1'
            if promo_class_label in metrics['Report']:
                st.metric(label="Precision", value=f"{metrics['Report'][promo_class_label]['precision']:.2f}", help="Of predicted promos, how many were correct?")
                st.metric(label="Recall", value=f"{metrics['Report'][promo_class_label]['recall']:.2f}", help="Of actual promos, how many did we catch?")
            else:
                st.info("No 'promo' events in test set for detailed metrics.")

        st.markdown("---")
        st.subheader("Strategic Recommendation for Upcoming Period")

        latest_our_data = df_our_product_pricing.tail(1)
        if not latest_our_data.empty:
            tomorrow_date = latest_our_data['Date'].iloc[0] + datetime.timedelta(days=1)
            
            prediction_features_df = pd.DataFrame([{
                'Our_Price': latest_our_data['Our_Price'].iloc[0],
                'Our_Inventory_Level': max(0, latest_our_data['Our_Inventory_Level'].iloc[0] - 5),
                'DayOfWeek': tomorrow_date.dayofweek,
                'Month': tomorrow_date.month,
                'Quarter': tomorrow_date.quarter,
                'Is_Our_Promo': 0
            }])
            
            prediction_features_processed = pd.get_dummies(prediction_features_df, columns=['DayOfWeek', 'Month', 'Quarter'], drop_first=True)
            for col in model_feature_names:
                if col not in prediction_features_processed.columns:
                    prediction_features_processed[col] = 0
            prediction_features_processed = prediction_features_processed[model_feature_names]

            tomorrow_promo_proba = model.predict_proba(prediction_features_processed)[:, 1][0]
            
            competitor_name = 'AcousticFlow Earbuds'
            
            if tomorrow_promo_proba > 0.55:
                st.error(
                    f"*CRITICAL ALERT!* There is a *{tomorrow_promo_proba:.1%} probability* that the key competitor *{competitor_name}* will launch a promotion soon.\n\n"
                    f"*Recommended Action:* *Launch a targeted counter-promotion or bundle offer.* Focus on the product's value to stress value over price."
                )
            else:
                st.success(
                    f"*Stable Market Forecast:* Low probability ({tomorrow_promo_proba:.1%}) of an immediate promotion from a key competitor.\n\n"
                    f"*Recommended Action:* *Optimize margins or invest in branding.* Use this stable period to strengthen your market position."
                )
        else:
            st.info("No historical data for our product to generate a future prediction.")

    st.markdown("---")
    st.subheader("Competitor Price Trend Analysis")

    competitor_cols = [col for col in df_pricing.columns if col.startswith('Competitor_Price_')]
    
    if competitor_cols:
        comparison_df = df_pricing[df_pricing['Product_SKU'] == selected_product_sku][['Date', 'Our_Price'] + competitor_cols].melt(
            id_vars=['Date'], var_name='Competitor', value_name='Price'
        )
        comparison_df['Competitor'] = comparison_df['Competitor'].str.replace('Competitor_Price_', '').str.replace('_', ' ')
        comparison_df['Competitor'] = comparison_df['Competitor'].apply(lambda x: "Our Product" if x == 'Our Price' else x)
        
        color_map = {'Our Product': '#e94560'}
        
        fig_comp_price = px.line(
            comparison_df,
            x='Date',
            y='Price',
            color='Competitor',
            title=f"Price Trend: {selected_product_sku} vs. Competitors",
            color_discrete_map=color_map,
            hover_data={'Price': ':.2f'}
        )
        fig_comp_price.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#e0e0e0",
                                     xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_comp_price, use_container_width=True)
    else:
        st.info("No competitor pricing data available for this product.")

    st.markdown("---")
    st.subheader("Raw Historical Pricing and Inventory Data")
    
    # Select key columns for the pricing table
    price_cols = [
        'Date', 
        'Our_Price', 
        'Our_Inventory_Level', 
        'Is_Our_Promo'
    ]
    # Dynamically add competitor price columns for the selected product
    comp_price_cols = [col for col in df_our_product_pricing.columns if col.startswith('Competitor_Price_')]
    
    display_pricing_df = df_our_product_pricing[price_cols + comp_price_cols].sort_values('Date', ascending=False).head(30)

    # Clean up column names for better display
    display_pricing_df.columns = [col.replace('_', ' ').title() for col in display_pricing_df.columns]
    
    st.dataframe(
        display_pricing_df,
        column_config={
            "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
            "Our Price": st.column_config.NumberColumn("Our Price", format="$%.2f"),
            "Our Inventory Level": st.column_config.NumberColumn("Our Inventory Level", format="%d units"),
            "Is Our Promo": st.column_config.CheckboxColumn("Is Our Promo", default=False),
        },
        hide_index=True,
        use_container_width=True
    )
