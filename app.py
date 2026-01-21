"""
AI Hope Wellness - Premium Health Intelligence Platform
Powered by LangChain | Created by The ATM Agency
Optimized for Streamlit Community Cloud Deployment

Features:
- LangChain document processing with FAISS vector store
- 20+ file format support (PDF, DOCX, CSV, images with OCR, etc.)
- 7 AI specialist consultants
- Comprehensive biomarker tracking and analysis
- Evidence-based wellness protocols
- Advanced health analytics and visualizations
"""

import streamlit as st
import streamlit_antd_components as sac
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from openai import OpenAI
import base64
import tempfile
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO, StringIO
import uuid

# --- LANGCHAIN IMPORTS (with graceful fallbacks) ---
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain, RetrievalQA
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chains.question_answering import load_qa_chain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Minimal Document class fallback
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

# --- DOCUMENT LOADERS (with fallbacks for Community Cloud) ---
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader,
        CSVLoader,
        UnstructuredExcelLoader,
        JSONLoader,
        UnstructuredHTMLLoader,
        UnstructuredMarkdownLoader,
        UnstructuredPowerPointLoader,
        UnstructuredXMLLoader,
    )
    DOCUMENT_LOADERS_AVAILABLE = True
except ImportError:
    DOCUMENT_LOADERS_AVAILABLE = False

# --- FALLBACK LOADERS ---
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# =====================================================
# BRANDING & CONFIGURATION
# =====================================================
BRAND_NAME = "AI Hope Wellness"
CREATED_BY = "The ATM Agency"
VERSION = "2.5.0"
PRIMARY_COLOR = "#16A34A"
SECONDARY_COLOR = "#22C55E"
ACCENT_COLOR = "#4ADE80"
TERTIARY_COLOR = "#86EFAC"
DANGER_COLOR = "#EF4444"
INFO_COLOR = "#15803D"
BG_COLOR = "#F0FDF4"
BG_SECONDARY = "#DCFCE7"

# =====================================================
# SUPPORTED FILE FORMATS (Enhanced)
# =====================================================
SUPPORTED_FORMATS = {
    "Documents": {
        "pdf": {"desc": "PDF Documents", "icon": "📄", "mime": "application/pdf"},
        "docx": {"desc": "Microsoft Word", "icon": "📝", "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        "doc": {"desc": "Microsoft Word (Legacy)", "icon": "📝", "mime": "application/msword"},
        "txt": {"desc": "Plain Text", "icon": "📃", "mime": "text/plain"},
        "rtf": {"desc": "Rich Text Format", "icon": "📋", "mime": "application/rtf"},
        "odt": {"desc": "OpenDocument Text", "icon": "📄", "mime": "application/vnd.oasis.opendocument.text"},
    },
    "Spreadsheets": {
        "csv": {"desc": "CSV Files", "icon": "📊", "mime": "text/csv"},
        "xlsx": {"desc": "Microsoft Excel", "icon": "📈", "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
        "xls": {"desc": "Excel (Legacy)", "icon": "📈", "mime": "application/vnd.ms-excel"},
        "tsv": {"desc": "Tab-Separated Values", "icon": "📊", "mime": "text/tab-separated-values"},
    },
    "Data Formats": {
        "json": {"desc": "JSON Data", "icon": "🔧", "mime": "application/json"},
        "xml": {"desc": "XML Data", "icon": "🔩", "mime": "application/xml"},
        "yaml": {"desc": "YAML Files", "icon": "⚙️", "mime": "text/yaml"},
        "yml": {"desc": "YAML Files", "icon": "⚙️", "mime": "text/yaml"},
    },
    "Web & Markup": {
        "html": {"desc": "HTML Pages", "icon": "🌐", "mime": "text/html"},
        "htm": {"desc": "HTML Pages", "icon": "🌐", "mime": "text/html"},
        "md": {"desc": "Markdown", "icon": "📑", "mime": "text/markdown"},
        "markdown": {"desc": "Markdown", "icon": "📑", "mime": "text/markdown"},
    },
    "Presentations": {
        "pptx": {"desc": "PowerPoint", "icon": "📽️", "mime": "application/vnd.openxmlformats-officedocument.presentationml.presentation"},
        "ppt": {"desc": "PowerPoint (Legacy)", "icon": "📽️", "mime": "application/vnd.ms-powerpoint"},
    },
    "Images (OCR via GPT-4 Vision)": {
        "png": {"desc": "PNG Images", "icon": "🖼️", "mime": "image/png"},
        "jpg": {"desc": "JPEG Images", "icon": "🖼️", "mime": "image/jpeg"},
        "jpeg": {"desc": "JPEG Images", "icon": "🖼️", "mime": "image/jpeg"},
        "webp": {"desc": "WebP Images", "icon": "🖼️", "mime": "image/webp"},
        "gif": {"desc": "GIF Images", "icon": "🖼️", "mime": "image/gif"},
        "bmp": {"desc": "Bitmap Images", "icon": "🖼️", "mime": "image/bmp"},
        "tiff": {"desc": "TIFF Images", "icon": "🖼️", "mime": "image/tiff"},
        "tif": {"desc": "TIFF Images", "icon": "🖼️", "mime": "image/tiff"},
        "heic": {"desc": "HEIC Images", "icon": "🖼️", "mime": "image/heic"},
    },
    "Health Data": {
        "hl7": {"desc": "HL7 Health Data", "icon": "🏥", "mime": "text/plain"},
        "ccda": {"desc": "CCDA Documents", "icon": "🏥", "mime": "application/xml"},
        "fhir": {"desc": "FHIR JSON", "icon": "🏥", "mime": "application/json"},
    }
}

ALL_EXTENSIONS = []
for category in SUPPORTED_FORMATS.values():
    ALL_EXTENSIONS.extend(category.keys())

IMAGE_EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tiff', 'tif', 'heic']

# =====================================================
# STREAMLIT PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title=f"{BRAND_NAME} | Premium Health Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/theatmagency/ai-hope-wellness',
        'Report a bug': 'https://github.com/theatmagency/ai-hope-wellness/issues',
        'About': f"""
        # {BRAND_NAME} v{VERSION}
        
        Premium AI-Powered Health Intelligence Platform
        
        Created by **{CREATED_BY}**
        
        Powered by LangChain, OpenAI GPT-4, and FAISS Vector Search
        """
    }
)

# =====================================================
# THEME CONSTANTS
# =====================================================
PRIMARY_COLOR = "#16A34A"
BG_COLOR = "#F0FDF4"
BG_SECONDARY = "#DCFCE7"
TEXT_COLOR = "#1a1a1a"
SECONDARY_COLOR = "#10B981"
ACCENT_COLOR = "#F59E0B"
TERTIARY_COLOR = "#D1FAE5"
INFO_COLOR = "#3B82F6"

# =====================================================
# PREMIUM UI STYLING (Enhanced)
# =====================================================
st.markdown(f"""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Base Styles */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}
    
    /* Hide Streamlit branding, GitHub, Fork, and top-right menu */
    #MainMenu {{visibility: hidden;}}
    /* header {{visibility: hidden;}} */
    
    /* Use native Streamlit sidebar toggle but ensure it matches the theme */
    [data-testid="stSidebarCollapse"] {{
        background-color: {BG_SECONDARY} !important;
        color: {PRIMARY_COLOR} !important;
        border-radius: 8px !important;
        border: 1px solid {PRIMARY_COLOR}30 !important;
    }}
    
    [data-testid="stSidebarCollapse"] svg {{
        fill: {PRIMARY_COLOR} !important;
    }}

    [data-testid="stToolbar"] {{display: none;}}
    [data-testid="stDecoration"] {{display: none;}}
    [data-testid="stStatusWidget"] {{display: none;}}
    .viewerBadge_container__r5tak {{display: none;}}
    .styles_viewerBadge__CvC9N {{display: none;}}
    ._profileContainer_gzau3_53 {{display: none;}}
    ._profilePreview_gzau3_63 {{display: none;}}
    [data-testid="manage-app-button"] {{display: none;}}
    
    /* App Background - Light Green Theme */
    .stApp {{
        background: linear-gradient(180deg, {BG_COLOR} 0%, {BG_SECONDARY} 25%, #ECFDF5 50%, {BG_SECONDARY} 75%, {BG_COLOR} 100%);
        background-attachment: fixed;
    }}
    
    /* Main Header Component */
    .main-header {{
        background: linear-gradient(135deg, white 0%, {BG_COLOR} 100%);
        padding: 2.5rem 3rem;
        border-radius: 28px;
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.08),
            0 0 0 1px rgba(0, 0, 0, 0.03),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        margin-bottom: 2.5rem;
        border-left: 8px solid {PRIMARY_COLOR};
        position: relative;
        overflow: hidden;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -100px;
        right: -100px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, {SECONDARY_COLOR}15, transparent 70%);
        border-radius: 50%;
    }}
    
    .main-header::after {{
        content: '';
        position: absolute;
        bottom: -50px;
        left: 50%;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, {PRIMARY_COLOR}08, transparent 70%);
        border-radius: 50%;
    }}
    
    /* Header Title Styles */
    .main-header h1 {{
        position: relative;
        z-index: 1;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }}
    
    .main-header h2 {{
        position: relative;
        z-index: 1;
        font-weight: 500;
        letter-spacing: -0.01em;
    }}
    
    .main-header p {{
        position: relative;
        z-index: 1;
    }}
    
    /* Badge Styles */
    .langchain-badge {{
        background: linear-gradient(135deg, {TERTIARY_COLOR}, {ACCENT_COLOR});
        color: #1a1a1a;
        padding: 0.5rem 1.25rem;
        border-radius: 24px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 12px {TERTIARY_COLOR}40;
    }}
    
    .version-badge {{
        background: linear-gradient(135deg, #0F172A, #1E293B);
        color: #94A3B8;
        padding: 0.35rem 0.75rem;
        border-radius: 12px;
        font-size: 0.65rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.75rem;
        letter-spacing: 0.03em;
    }}
    
    .status-badge {{
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }}
    
    .status-badge.success {{
        background: #D1FAE5;
        color: #065F46;
    }}
    
    .status-badge.warning {{
        background: #FEF3C7;
        color: #92400E;
    }}
    
    .status-badge.danger {{
        background: #FEE2E2;
        color: #991B1B;
    }}
    
    .status-badge.info {{
        background: #DBEAFE;
        color: #1E40AF;
    }}
    
    /* Agency Footer */
    .agency-footer {{
        text-align: center;
        padding: 3rem 2rem;
        color: #64748B;
        font-size: 0.9rem;
        border-top: 2px solid #E2E8F0;
        margin-top: 5rem;
        background: linear-gradient(180deg, white 0%, #F8FAFC 100%);
        border-radius: 28px 28px 0 0;
        box-shadow: 0 -10px 40px rgba(0, 0, 0, 0.05);
    }}
    
    /* Wellness Card Component */
    .wellness-card {{
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 
            0 4px 6px -1px rgba(0, 0, 0, 0.07),
            0 2px 4px -1px rgba(0, 0, 0, 0.04),
            0 0 0 1px rgba(0, 0, 0, 0.02);
        border: 1px solid #E2E8F0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .wellness-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}, {ACCENT_COLOR});
        opacity: 0;
        transition: opacity 0.4s ease;
    }}
    
    .wellness-card:hover {{
        transform: translateY(-8px);
        box-shadow: 
            0 25px 50px -12px rgba(0, 0, 0, 0.12),
            0 0 0 1px rgba(16, 185, 129, 0.1);
    }}
    
    .wellness-card:hover::before {{
        opacity: 1;
    }}
    
    /* Stat Card Component */
    .stat-card {{
        background: linear-gradient(135deg, white 0%, #F8FAFC 100%);
        padding: 1.75rem;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .stat-card:hover {{
        border-color: {SECONDARY_COLOR};
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15);
    }}
    
    .stat-value {{
        font-size: 2.75rem;
        font-weight: 900;
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }}
    
    .stat-label {{
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }}
    
    .stat-delta {{
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.25rem 0.5rem;
        border-radius: 8px;
        display: inline-block;
        margin-top: 0.5rem;
    }}
    
    .stat-delta.positive {{
        background: #D1FAE5;
        color: #065F46;
    }}
    
    .stat-delta.negative {{
        background: #FEE2E2;
        color: #991B1B;
    }}
    
    /* Format Display Grid */
    .format-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }}
    
    .format-chip {{
        background: linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 100%);
        padding: 0.75rem 1rem;
        border-radius: 12px;
        font-size: 0.8rem;
        text-align: center;
        border: 1px solid #E2E8F0;
        transition: all 0.3s ease;
        cursor: default;
    }}
    
    .format-chip:hover {{
        background: linear-gradient(135deg, {SECONDARY_COLOR}15, {PRIMARY_COLOR}10);
        border-color: {SECONDARY_COLOR};
        transform: translateY(-2px);
    }}
    
    /* Processing Step Component */
    .processing-step {{
        display: flex;
        align-items: center;
        gap: 1.25rem;
        padding: 1.25rem 1.5rem;
        background: linear-gradient(135deg, #F8FAFC 0%, white 100%);
        border-radius: 16px;
        margin: 0.75rem 0;
        border-left: 4px solid {SECONDARY_COLOR};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }}
    
    .processing-step:hover {{
        background: linear-gradient(135deg, white 0%, #F8FAFC 100%);
        transform: translateX(4px);
    }}
    
    .processing-step .step-number {{
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, {ACCENT_COLOR}, {TERTIARY_COLOR});
        color: #1a1a1a;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
        flex-shrink: 0;
    }}
    
    /* Sidebar Styling - Light Green Theme */
    [data-testid="stSidebar"] {{
        background-color: {BG_SECONDARY} !important;
        background-image: none !important;
        border-right: 1px solid {PRIMARY_COLOR}20 !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: {TEXT_COLOR} !important;
    }}
    
    [data-testid="stSidebar"] .stRadio > label {{
        color: {TEXT_COLOR} !important;
    }}
    
    [data-testid="stSidebar"] hr {{
        border-color: {SECONDARY_COLOR};
    }}
    
    /* Sidebar Brand */
    .sidebar-brand {{
        padding: 1.5rem 0;
        border-bottom: 1px solid {SECONDARY_COLOR};
        margin-bottom: 1.5rem;
    }}
    
    .sidebar-brand h1 {{
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #1a1a1a 0%, #374151 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    /* Button Styles - Green Theme */
    .stButton>button {{
        background: linear-gradient(135deg, {ACCENT_COLOR} 0%, {TERTIARY_COLOR} 100%);
        color: #1a1a1a;
        border-radius: 14px;
        border: none;
        font-weight: 700;
        padding: 0.85rem 2.25rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px {PRIMARY_COLOR}30;
        font-size: 0.95rem;
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, {INFO_COLOR} 0%, {PRIMARY_COLOR} 100%);
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(22, 163, 74, 0.35);
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, {SECONDARY_COLOR} 0%, #059669 100%);
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(16, 185, 129, 0.35);
    }}
    
    .stButton>button:active {{
        transform: translateY(-1px);
    }}
    
    /* Document Preview */
    .document-preview {{
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #E2E8F0;
        padding: 2rem;
        border-radius: 16px;
        font-family: 'JetBrains Mono', 'Monaco', 'Consolas', monospace;
        font-size: 0.85rem;
        max-height: 450px;
        overflow-y: auto;
        line-height: 1.7;
        border: 1px solid #334155;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.2);
    }}
    
    .document-preview::-webkit-scrollbar {{
        width: 8px;
    }}
    
    .document-preview::-webkit-scrollbar-track {{
        background: #1E293B;
        border-radius: 4px;
    }}
    
    .document-preview::-webkit-scrollbar-thumb {{
        background: #475569;
        border-radius: 4px;
    }}
    
    /* Chunk Card */
    .chunk-card {{
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0;
        border-left: 5px solid {ACCENT_COLOR};
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }}
    
    .chunk-card:hover {{
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.15);
        transform: translateX(4px);
    }}
    
    /* Vector Visualization */
    .vector-viz {{
        background: linear-gradient(135deg, {BG_COLOR}, {BG_SECONDARY});
        padding: 2.5rem;
        border-radius: 20px;
        color: #1a1a1a;
        border: 1px solid {SECONDARY_COLOR};
    }}
    
    /* Specialist Card */
    .specialist-card {{
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        border: 1px solid #E2E8F0;
        transition: all 0.3s ease;
        cursor: pointer;
    }}
    
    .specialist-card:hover {{
        border-color: {SECONDARY_COLOR};
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.12);
        transform: translateY(-3px);
    }}
    
    .specialist-card.active {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 8px 25px rgba(30, 58, 138, 0.2);
        background: linear-gradient(135deg, white 0%, {PRIMARY_COLOR}05 100%);
    }}
    
    /* Chat Message Styles */
    [data-testid="stChatMessage"] {{
        background: white;
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }}
    
    /* Metric Cards */
    [data-testid="metric-container"] {{
        background: white;
        padding: 1.25rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
    }}
    
    /* Tab Styling */
        /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.75rem;
        background: #F1F5F9;
        padding: 0.75rem;
        border-radius: 20px;
        overflow-x: auto;
        scrollbar-width: none; /* Firefox */
    }}
    
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
        display: none; /* Chrome, Safari, Opera */
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 14px;
        padding: 1rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid transparent;
        white-space: nowrap;
        min-width: 120px;
        justify-content: center;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(255, 255, 255, 0.5);
        transform: translateY(-2px);
    }}
    
    .stTabs [aria-selected="true"] {{
        background: white !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: {PRIMARY_COLOR}40 !important;
        color: {PRIMARY_COLOR} !important;
    }}

    /* Mobile Responsiveness for Tabs */
    @media (max-width: 768px) {{
        .stTabs [data-baseweb="tab"] {{
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
            min-width: 100px;
        }}
        .main-header {{
            padding: 1.5rem;
        }}
        .main-header h1 {{
            font-size: 1.75rem !important;
        }}
    }}
    
    /* Collapsible Sidebar Improvements */
    [data-testid="stSidebarNav"] {{
        background-image: none;
    }}
    
    /* Make sidebar radio buttons look like big buttons */
    div[data-testid="stSidebar"] .stRadio > div {{
        gap: 0.5rem;
    }}
    
    div[data-testid="stSidebar"] .stRadio label {{
        background: white !important;
        color: {TEXT_COLOR} !important;
        padding: 1rem 1.25rem !important;
        border-radius: 12px !important;
        border: 1px solid {PRIMARY_COLOR}20 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        cursor: pointer !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        display: flex !important;
        align-items: center !important;
    }}
    
    div[data-testid="stSidebar"] .stRadio label:hover {{
        background: {BG_COLOR} !important;
        border-color: {PRIMARY_COLOR} !important;
        transform: translateX(3px) !important;
    }}
    
    div[data-testid="stSidebar"] .stRadio label[data-active="true"] {{
        background: {PRIMARY_COLOR} !important;
        color: white !important;
        border-color: {PRIMARY_COLOR} !important;
        font-weight: 800 !important;
        box-shadow: 0 8px 16px {PRIMARY_COLOR}40 !important;
    }}
    
    /* Target the selected radio button specifically in Streamlit's DOM */
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {{
        display: none; /* Hide the actual radio circle */
    }}
    
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {{
        margin-bottom: 4px;
        border-radius: 10px;
    }}
    
    /* Navigation Expander Styling */
    div[data-testid="stSidebar"] .streamlit-expanderHeader {{
        background: white !important;
        color: {TEXT_COLOR} !important;
        border-radius: 12px !important;
        border: 1px solid {PRIMARY_COLOR}30 !important;
        padding: 0.75rem 1rem !important;
        font-weight: 700 !important;
    }}
    
    div[data-testid="stSidebar"] .streamlit-expanderContent {{
        background: transparent !important;
        border: none !important;
        padding-top: 1rem !important;
    }}
    
    /* Expander Styling */
    .streamlit-expanderHeader {{
        background: white;
        border-radius: 12px;
        font-weight: 600;
    }}
    
    /* Input Styling */
    .stTextInput>div>div>input {{
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        padding: 0.75rem 1rem;
        transition: all 0.3s ease;
    }}
    
    .stTextInput>div>div>input:focus {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 0 3px {PRIMARY_COLOR}20;
    }}
    
    /* Progress Bar */
    .stProgress > div > div {{
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        border-radius: 10px;
    }}
    
    /* Protocol Card */
    .protocol-card {{
        background: linear-gradient(135deg, white 0%, #F8FAFC 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #E2E8F0;
        position: relative;
        overflow: hidden;
    }}
    
    .protocol-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
    }}
    
    .protocol-card h3 {{
        color: {PRIMARY_COLOR};
        margin-bottom: 1rem;
    }}
    
    /* Biomarker Row */
    .biomarker-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.5rem;
        background: white;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid #E2E8F0;
        transition: all 0.3s ease;
    }}
    
    .biomarker-row:hover {{
        background: #F8FAFC;
        transform: translateX(4px);
    }}
    
    /* Animation Keyframes */
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .animate-pulse {{
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }}
    
    .animate-slide-in {{
        animation: slideIn 0.5s ease-out;
    }}
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: #F1F5F9;
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, #CBD5E1, #94A3B8);
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, #94A3B8, #64748B);
    }}
    
    /* Quick Action Button Grid */
    .quick-action-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }}
    
    .quick-action-btn {{
        background: linear-gradient(135deg, #F8FAFC 0%, white 100%);
        border: 2px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 600;
        color: #475569;
    }}
    
    .quick-action-btn:hover {{
        border-color: {SECONDARY_COLOR};
        background: linear-gradient(135deg, {SECONDARY_COLOR}10, white);
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.15);
    }}
    
    /* Health Score Ring */
    .health-score-ring {{
        position: relative;
        width: 200px;
        height: 200px;
        margin: 0 auto;
    }}
    
    .health-score-value {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
    }}
    
    .health-score-value .score {{
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }}
    
    .health-score-value .label {{
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.25rem;
    }}
    
    /* Horizontal Navigation Menu */
    .nav-container {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 0.75rem 2rem;
        border-bottom: 1px solid {PRIMARY_COLOR}20;
        margin: -6rem -5rem 2rem -5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }
    
    .nav-logo {
        font-weight: 800;
        font-size: 1.25rem;
        color: {PRIMARY_COLOR};
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .nav-items {
        display: flex;
        gap: 0.5rem;
        overflow-x: auto;
        scrollbar-width: none;
    }
    
    .nav-items::-webkit-scrollbar {
        display: none;
    }
    
    .nav-item {
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #475569;
        cursor: pointer;
        transition: all 0.2s ease;
        white-space: nowrap;
        text-decoration: none;
        border: 1px solid transparent;
    }
    
    .nav-item:hover {
        background: {BG_COLOR};
        color: {PRIMARY_COLOR};
    }
    
    .nav-item.active {
        background: {PRIMARY_COLOR};
        color: white;
        box-shadow: 0 4px 12px {PRIMARY_COLOR}40;
    }

    /* Responsive Adjustments */
    @media (max-width: 768px) {
        .nav-container {
            margin: -6rem -1rem 1rem -1rem;
            padding: 0.75rem 1rem;
            flex-direction: column;
            gap: 0.75rem;
        }
        .nav-items {
            width: 100%;
            justify-content: flex-start;
        }
        .main-header {
            padding: 1.5rem;
            border-radius: 20px;
        }
        .main-header h1 {
            font-size: 1.5rem;
        }
        .stat-value {
            font-size: 2rem;
        }
        .format-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# EXPANDED CLINICAL SPECIALIST REGISTRY
# =====================================================
SPECIALISTS = {
    "Chief Medical Officer": {
        "icon": "👨‍⚕️",
        "bio": "Expert in integrative medicine, clinical strategy, and executive health optimization. Board-certified in Internal Medicine with fellowship training in Functional Medicine.",
        "color": PRIMARY_COLOR,
        "expertise": ["Clinical Strategy", "Integrative Medicine", "Preventive Care", "Executive Health"],
        "prompt": """You are the Chief Medical Officer of AI Hope Wellness, a premium health intelligence platform created by The ATM Agency.

ROLE & EXPERTISE:
- You are a board-certified physician with expertise in integrative and functional medicine
- You provide high-level, strategic health guidance integrating all aspects of longevity, metabolic health, and preventive care
- You coordinate care recommendations across all specialist domains
- You have access to the patient's uploaded documents and lab results through our LangChain-powered document analysis system

COMMUNICATION STYLE:
- Maintain a premium, clinical, and supportive tone
- Structure responses with clear headers and bullet points
- Provide evidence-based recommendations with relevant citations
- Balance thoroughness with accessibility

RESPONSE STRUCTURE:
1. Clinical Assessment Summary
2. Key Findings & Observations
3. Strategic Recommendations
4. Priority Action Items
5. Suggested Follow-up

Always include a comprehensive medical disclaimer at the end of your response."""
    },
    "Peptide & Bio-Regulator Expert": {
        "icon": "🧬",
        "bio": "World-leading specialist in therapeutic peptides, bio-regulators, and cellular signaling pathways. Extensive research background in regenerative medicine.",
        "color": SECONDARY_COLOR,
        "expertise": ["Tissue Repair Peptides", "GH Secretagogues", "Immune Modulators", "Longevity Peptides"],
        "prompt": """You are a world-leading Peptide and Bio-Regulator Specialist at AI Hope Wellness, created by The ATM Agency.

AREAS OF EXPERTISE:

TISSUE REPAIR & RECOVERY:
- BPC-157 (Body Protection Compound): Gut healing, tendon repair, neuroprotection
- TB-500 (Thymosin Beta-4): Systemic tissue repair, anti-inflammatory
- GHK-Cu (Copper Peptide): Skin health, wound healing, anti-aging collagen synthesis
- KPV: Anti-inflammatory, gut healing, skin conditions

GROWTH HORMONE OPTIMIZATION:
- CJC-1295/Ipamorelin: GH pulse optimization, body composition
- Tesamorelin: Visceral fat reduction, cognitive support
- MK-677 (Ibutamoren): Oral GH secretagogue, sleep quality
- Sermorelin: Gentle GH stimulation, anti-aging

IMMUNE MODULATION:
- Thymosin Alpha-1: Immune enhancement, anti-viral, cancer support
- LL-37: Antimicrobial, wound healing, immune modulation

LONGEVITY & CELLULAR HEALTH:
- Epitalon: Telomerase activation, pineal gland support
- FOXO4-DRI: Senolytic peptide, cellular clearing
- SS-31 (Elamipretide): Mitochondrial support, cardioprotection
- MOTS-c: Metabolic regulation, exercise mimetic

RESPONSE GUIDELINES:
- Discuss mechanisms of action at a sophisticated level
- Provide dosing ranges and administration routes
- Address timing, cycling, and stacking strategies
- Cover reconstitution, storage, and quality considerations
- Emphasize safety, proper sourcing, and medical supervision
- Include relevant research citations when available

Always include a disclaimer about the experimental nature of many peptides and the need for medical supervision."""
    },
    "Hormone Optimization MD": {
        "icon": "⚖️",
        "bio": "Senior Endocrinologist specializing in hormone optimization, TRT, HRT, and comprehensive metabolic-hormonal integration for peak performance.",
        "color": ACCENT_COLOR,
        "expertise": ["TRT/HRT", "Thyroid Optimization", "Adrenal Health", "Metabolic Hormones"],
        "prompt": """You are a Senior Endocrinologist specializing in hormone optimization at AI Hope Wellness, created by The ATM Agency.

AREAS OF EXPERTISE:

MALE HORMONE OPTIMIZATION:
- Testosterone Replacement Therapy (TRT): Protocols, monitoring, optimization
- HCG: Fertility preservation, testicular function
- Clomiphene/Enclomiphene: HPTA preservation, alternative to TRT
- Anastrozole: Estrogen management, E2 optimization

FEMALE HORMONE OPTIMIZATION:
- Hormone Replacement Therapy (HRT): Bioidentical protocols
- Progesterone: Timing, forms, benefits
- DHEA: Androgen precursor, energy, cognition
- Testosterone (women): Low-dose benefits, considerations

THYROID OPTIMIZATION:
- T4/T3 ratios and conversion
- Reverse T3 and thyroid resistance
- Thyroid antibodies and autoimmunity
- Iodine, selenium, zinc for thyroid health

ADRENAL & STRESS HORMONES:
- Cortisol patterns and HPA axis
- DHEA-S and pregnenolone
- Adrenal fatigue vs. dysfunction
- Stress resilience protocols

METABOLIC HORMONES:
- Insulin sensitivity and resistance
- GH/IGF-1 axis
- Leptin and ghrelin
- Adiponectin

KEY PRINCIPLES:
- Focus on OPTIMAL ranges, not just reference ranges
- Address the interplay between different hormone systems
- Consider the whole patient, not just lab numbers
- Emphasize regular monitoring and adjustment
- Discuss lifestyle factors that impact hormones

Always provide protocols with appropriate monitoring markers and include medical disclaimer."""
    },
    "Functional Nutritionist": {
        "icon": "🥗",
        "bio": "Specialist in nutrigenomics, functional medicine nutrition, gut-brain axis optimization, and metabolic flexibility protocols.",
        "color": "#EC4899",
        "expertise": ["Nutrigenomics", "Gut Health", "Metabolic Nutrition", "Therapeutic Diets"],
        "prompt": """You are a Functional Nutritionist at AI Hope Wellness, created by The ATM Agency.

AREAS OF EXPERTISE:

NUTRIGENOMICS & PERSONALIZED NUTRITION:
- MTHFR and methylation support
- APOE considerations for fat and cholesterol
- FTO and weight management genetics
- Caffeine, alcohol, and detox genetics

GUT MICROBIOME OPTIMIZATION:
- Microbiome testing interpretation
- Prebiotic and probiotic strategies
- SIBO and dysbiosis protocols
- Leaky gut and intestinal permeability
- Gut-brain axis optimization

METABOLIC FLEXIBILITY:
- Ketogenic and low-carb strategies
- Carb cycling for performance
- Metabolic switching and fasting
- Glucose optimization strategies
- Continuous glucose monitor interpretation

THERAPEUTIC DIETS:
- Anti-inflammatory protocols
- Elimination diets (AIP, low FODMAP, low histamine)
- Carnivore and animal-based approaches
- Mediterranean and Blue Zones principles

TARGETED SUPPLEMENTATION:
- Foundational supplements by life stage
- Condition-specific protocols
- Bioavailability and timing considerations
- Quality and sourcing guidance

RESPONSE GUIDELINES:
- Provide specific food recommendations
- Include meal timing and composition guidance
- Address supplement protocols with dosing
- Consider individual variation and testing
- Explain the "why" behind recommendations

Always include disclaimer about individual needs and medical supervision."""
    },
    "Performance & Longevity Coach": {
        "icon": "⚡",
        "bio": "Expert in VO2 Max optimization, strength training for longevity, biological age reversal, and evidence-based anti-aging interventions.",
        "color": "#06B6D4",
        "expertise": ["VO2 Max", "Strength Training", "Biological Age", "Recovery Optimization"],
        "prompt": """You are a Performance and Longevity Coach at AI Hope Wellness, created by The ATM Agency.

AREAS OF EXPERTISE:

CARDIORESPIRATORY FITNESS:
- VO2 Max testing and improvement strategies
- Zone 2 training for metabolic health
- High-intensity interval protocols
- Cardio programming for longevity
- Heart rate variability optimization

STRENGTH & MUSCLE:
- Resistance training for longevity
- Muscle protein synthesis optimization
- Sarcopenia prevention
- Functional movement patterns
- Load management and progression

BIOLOGICAL AGE REVERSAL:
- Epigenetic age testing interpretation
- Interventions that move biological age markers
- DNA methylation optimization
- Telomere length considerations
- Cellular senescence strategies

RECOVERY & ADAPTATION:
- Sleep optimization for recovery
- Stress management and HRV
- Active recovery protocols
- Deload and periodization strategies

HORMETIC STRESSORS:
- Sauna protocols and heat shock proteins
- Cold exposure and brown fat activation
- Hypoxia training considerations
- Fasting and autophagy

LONGEVITY EXERCISE FRAMEWORK:
- The "Centenarian Decathlon" concept
- Maintaining function into old age
- Balance, mobility, and fall prevention
- Grip strength and its mortality correlation

Provide specific, actionable protocols with progressions and emphasize sustainable, evidence-based approaches. Include disclaimer."""
    },
    "Sleep & Circadian Specialist": {
        "icon": "🌙",
        "bio": "Expert in sleep architecture optimization, circadian rhythm entrainment, chronobiology, and sleep disorders management.",
        "color": "#6366F1",
        "expertise": ["Sleep Architecture", "Circadian Rhythms", "Chronotherapy", "Sleep Disorders"],
        "prompt": """You are a Sleep and Circadian Specialist at AI Hope Wellness, created by The ATM Agency.

AREAS OF EXPERTISE:

SLEEP ARCHITECTURE:
- Sleep stages and their functions
- Deep sleep optimization (N3)
- REM sleep enhancement
- Sleep efficiency metrics
- Sleep tracker interpretation

CIRCADIAN RHYTHM OPTIMIZATION:
- Light exposure timing and intensity
- Meal timing and circadian alignment
- Temperature rhythm management
- Chronotype assessment (early bird vs. night owl)
- Social jetlag and its impacts

SLEEP HORMONES & NEUROCHEMISTRY:
- Melatonin: timing, dosing, forms
- Adenosine and sleep pressure
- GABA and sleep initiation
- Orexin/hypocretin system
- Cortisol awakening response

SLEEP DISORDERS:
- Insomnia types and management
- Sleep apnea screening and optimization
- Restless legs syndrome
- Circadian rhythm disorders
- Parasomnias

SLEEP OPTIMIZATION PROTOCOLS:

Evening Protocol (3-hour runway):
- Light management
- Temperature manipulation
- Food and beverage timing
- Wind-down routines

Supplement Stack:
- Magnesium (forms and timing)
- L-theanine, Apigenin, Glycine
- GABA precursors
- Melatonin (when appropriate)

Environment Optimization:
- Temperature (65-68°F)
- Darkness (blackout)
- Sound management
- Air quality

Morning Protocol:
- Light exposure timing
- Consistent wake times
- Caffeine timing

Provide specific, evidence-based protocols with timing and dosing. Address the bidirectional relationship between sleep and all health markers. Include disclaimer."""
    },
    "Mental Health & Neuroptimization": {
        "icon": "🧠",
        "bio": "Specialist in cognitive enhancement, mood optimization, neurotransmitter balance, neuroplasticity, and brain health protocols.",
        "color": "#8B5CF6",
        "expertise": ["Cognitive Enhancement", "Mood Optimization", "Neuroplasticity", "Brain Health"],
        "prompt": """You are a Mental Health and Neuroptimization Specialist at AI Hope Wellness, created by The ATM Agency.

AREAS OF EXPERTISE:

COGNITIVE ENHANCEMENT:
- Attention and focus optimization
- Memory enhancement strategies
- Processing speed improvement
- Executive function support
- Learning and skill acquisition

NEUROTRANSMITTER OPTIMIZATION:
- Dopamine: motivation, reward, focus
- Serotonin: mood, sleep, gut health
- Acetylcholine: memory, cognition
- GABA: calm, sleep, anxiety
- Norepinephrine: alertness, attention

NOOTROPIC PROTOCOLS:

Foundation Stack:
- Lion's Mane: NGF, BDNF support
- Omega-3 DHA: neuronal membranes
- Phosphatidylserine: cell signaling
- Bacopa Monnieri: memory consolidation

Acute Performance:
- Alpha-GPC: acetylcholine precursor
- Caffeine + L-Theanine: focused energy
- Rhodiola Rosea: stress resilience
- Modafinil (prescription): wakefulness

Neuroprotection:
- Curcumin (Longvida): anti-inflammatory
- Blueberry/anthocyanins: vascular health
- Cocoa flavanols: blood flow
- Resveratrol: sirtuin activation

NEUROPLASTICITY & BDNF:
- Exercise protocols for BDNF
- Fasting and ketosis effects
- Cold exposure benefits
- Novel learning stimulation
- Social connection importance

MOOD OPTIMIZATION:
- Depression support strategies
- Anxiety management protocols
- Stress resilience building
- HPA axis regulation
- Gut-brain axis optimization

BRAIN HEALTH LONGEVITY:
- Alzheimer's prevention strategies
- Vascular cognitive health
- Inflammation and neurodegeneration
- Sleep and glymphatic clearance

Address the mind-body connection in all recommendations. Provide specific protocols with mechanisms. Include appropriate disclaimers about mental health conditions requiring professional care."""
    }
}

# =====================================================
# COMPREHENSIVE BIOMARKER REFERENCE DATABASE
# =====================================================
BIOMARKER_DATABASE = {
    "Metabolic Health": {
        "HbA1c": {
            "unit": "%",
            "optimal": "4.5-5.2",
            "standard": "4.0-5.6",
            "concern": ">5.7 (prediabetes), >6.4 (diabetes)",
            "description": "3-month average blood sugar; gold standard for glycemic control",
            "factors": ["Diet", "Exercise", "Sleep", "Stress", "Medications"]
        },
        "Fasting Glucose": {
            "unit": "mg/dL",
            "optimal": "75-90",
            "standard": "70-100",
            "concern": ">100 (prediabetes), >126 (diabetes)",
            "description": "Blood sugar after overnight fast; insulin resistance marker",
            "factors": ["Last meal timing", "Sleep quality", "Stress", "Dawn phenomenon"]
        },
        "Fasting Insulin": {
            "unit": "uIU/mL",
            "optimal": "2-6",
            "standard": "2-25",
            "concern": ">10 (early insulin resistance)",
            "description": "Insulin levels after fasting; sensitive IR marker",
            "factors": ["Carb intake", "Body composition", "Exercise", "Sleep"]
        },
        "HOMA-IR": {
            "unit": "ratio",
            "optimal": "<1.0",
            "standard": "<2.5",
            "concern": ">2.5 (insulin resistant)",
            "description": "Calculated insulin resistance index",
            "factors": ["Glucose x Insulin / 405"]
        },
        "Triglycerides": {
            "unit": "mg/dL",
            "optimal": "<75",
            "standard": "<150",
            "concern": ">150 (elevated), >500 (high risk)",
            "description": "Blood fats; carb/alcohol sensitive metabolic marker",
            "factors": ["Carb intake", "Alcohol", "Omega-3s", "Exercise"]
        }
    },
    "Lipid Panel": {
        "Total Cholesterol": {
            "unit": "mg/dL",
            "optimal": "150-200",
            "standard": "<200",
            "concern": ">240 (elevated)",
            "description": "Total blood cholesterol; less useful in isolation",
            "factors": ["Diet", "Genetics", "Thyroid", "Medications"]
        },
        "LDL Cholesterol": {
            "unit": "mg/dL",
            "optimal": "<100",
            "standard": "<130",
            "concern": ">160 (elevated)",
            "description": "'Bad' cholesterol; cardiovascular risk marker",
            "factors": ["Saturated fat", "Genetics", "Inflammation", "Particle size matters"]
        },
        "HDL Cholesterol": {
            "unit": "mg/dL",
            "optimal": ">60",
            "standard": ">40 (M), >50 (F)",
            "concern": "<40 (low)",
            "description": "'Good' cholesterol; reverse cholesterol transport",
            "factors": ["Exercise", "Omega-3s", "Alcohol (moderate)", "Genetics"]
        },
        "LDL Particle Number": {
            "unit": "nmol/L",
            "optimal": "<1000",
            "standard": "<1300",
            "concern": ">1600 (high risk)",
            "description": "Actual LDL particle count; more predictive than LDL-C",
            "factors": ["Insulin resistance", "Inflammation", "Genetics"]
        },
        "Lp(a)": {
            "unit": "nmol/L",
            "optimal": "<75",
            "standard": "<75",
            "concern": ">75 (elevated genetic risk)",
            "description": "Genetic cardiovascular risk marker; largely unchangeable",
            "factors": ["Genetics (90%)", "Niacin may help", "PCSK9 inhibitors"]
        },
        "ApoB": {
            "unit": "mg/dL",
            "optimal": "<80",
            "standard": "<100",
            "concern": ">130 (elevated)",
            "description": "Total atherogenic particle count; excellent risk marker",
            "factors": ["Diet", "Medications", "Genetics"]
        }
    },
    "Inflammation": {
        "hs-CRP": {
            "unit": "mg/L",
            "optimal": "<0.5",
            "standard": "<1.0",
            "concern": ">3.0 (high cardiovascular risk)",
            "description": "Systemic inflammation marker; cardiovascular risk",
            "factors": ["Diet", "Sleep", "Exercise", "Infections", "Obesity"]
        },
        "Homocysteine": {
            "unit": "umol/L",
            "optimal": "<7",
            "standard": "5-15",
            "concern": ">12 (elevated)",
            "description": "Methylation and cardiovascular marker",
            "factors": ["B vitamins (B12, folate, B6)", "MTHFR status", "Kidney function"]
        },
        "Ferritin": {
            "unit": "ng/mL",
            "optimal": "40-100 (M), 30-80 (F)",
            "standard": "30-400 (M), 20-200 (F)",
            "concern": ">300 (may indicate iron overload or inflammation)",
            "description": "Iron storage protein; also acute phase reactant",
            "factors": ["Iron intake", "Blood donation", "Inflammation", "Liver health"]
        },
        "ESR": {
            "unit": "mm/hr",
            "optimal": "<10",
            "standard": "<20 (M), <30 (F)",
            "concern": ">30 (elevated inflammation)",
            "description": "Non-specific inflammation marker",
            "factors": ["Age", "Infections", "Autoimmune conditions"]
        },
        "Uric Acid": {
            "unit": "mg/dL",
            "optimal": "<5.5",
            "standard": "3.5-7.2 (M), 2.5-6.2 (F)",
            "concern": ">7.0 (gout risk, metabolic dysfunction)",
            "description": "Purine metabolism; marker of metabolic health",
            "factors": ["Fructose", "Alcohol", "Purines", "Kidney function"]
        }
    },
    "Hormones - Male": {
        "Total Testosterone": {
            "unit": "ng/dL",
            "optimal": "700-900",
            "standard": "300-1000",
            "concern": "<400 (suboptimal), <300 (deficient)",
            "description": "Primary male androgen; energy, muscle, mood, libido",
            "factors": ["Age", "Sleep", "Stress", "Body fat", "Exercise"]
        },
        "Free Testosterone": {
            "unit": "pg/mL",
            "optimal": "15-25",
            "standard": "5-25",
            "concern": "<10 (low bioavailable T)",
            "description": "Unbound, bioavailable testosterone",
            "factors": ["SHBG levels", "Total T", "Albumin"]
        },
        "SHBG": {
            "unit": "nmol/L",
            "optimal": "20-40",
            "standard": "10-70",
            "concern": ">50 (less free T), <20 (may indicate insulin resistance)",
            "description": "Sex hormone binding globulin; modulates free hormones",
            "factors": ["Thyroid", "Insulin", "Liver", "Estrogen"]
        },
        "Estradiol (E2)": {
            "unit": "pg/mL",
            "optimal": "20-35",
            "standard": "10-40",
            "concern": ">50 (high, may need AI), <15 (may be too low)",
            "description": "Primary estrogen; needed for libido, bone, brain but excess problematic",
            "factors": ["Body fat (aromatase)", "Testosterone levels", "AI use"]
        },
        "LH": {
            "unit": "mIU/mL",
            "optimal": "3-8",
            "standard": "1.5-9.3",
            "concern": "Elevated with low T = primary hypogonadism; Low with low T = secondary",
            "description": "Luteinizing hormone; stimulates testosterone production",
            "factors": ["Pituitary function", "Feedback from T/E2"]
        },
        "FSH": {
            "unit": "mIU/mL",
            "optimal": "2-8",
            "standard": "1.5-12.4",
            "concern": "Elevated may indicate testicular dysfunction",
            "description": "Follicle-stimulating hormone; spermatogenesis",
            "factors": ["Pituitary function", "Testicular health"]
        },
        "Prolactin": {
            "unit": "ng/mL",
            "optimal": "5-15",
            "standard": "4-15",
            "concern": ">20 (may suppress T, libido)",
            "description": "Elevated can suppress testosterone and libido",
            "factors": ["Stress", "Sleep", "Medications", "Pituitary tumors"]
        },
        "DHT": {
            "unit": "ng/dL",
            "optimal": "30-85",
            "standard": "30-85",
            "concern": "High may contribute to hair loss, prostate issues",
            "description": "Dihydrotestosterone; potent androgen from T conversion",
            "factors": ["5-alpha reductase activity", "Testosterone levels"]
        }
    },
    "Hormones - Female": {
        "Estradiol (E2)": {
            "unit": "pg/mL",
            "optimal": "Varies by cycle phase and menopause status",
            "standard": "Follicular: 12-165, Ovulation: 85-500, Luteal: 43-210, Post-menopause: <10-20",
            "concern": "Low in menopause causes symptoms; high may increase certain cancer risks",
            "description": "Primary estrogen; bone, brain, cardiovascular, skin health",
            "factors": ["Menstrual cycle phase", "Age", "Body fat", "HRT status"]
        },
        "Progesterone": {
            "unit": "ng/mL",
            "optimal": "Luteal phase: 10-20",
            "standard": "Luteal: 2-25, Post-menopause: <0.5",
            "concern": "Low luteal progesterone = possible infertility, PMS",
            "description": "Calming hormone; sleep, mood, bone, pregnancy",
            "factors": ["Ovulation", "Stress (progesterone steal)", "Age"]
        },
        "Testosterone (Total)": {
            "unit": "ng/dL",
            "optimal": "40-70",
            "standard": "15-70",
            "concern": "<25 (low energy, libido), >70 (may indicate PCOS)",
            "description": "Important for women too - energy, libido, muscle, mood",
            "factors": ["Age", "Adrenal function", "PCOS", "HRT"]
        },
        "DHEA-S": {
            "unit": "ug/dL",
            "optimal": "200-350",
            "standard": "Age-dependent, declines with age",
            "concern": "Low associated with fatigue, low libido",
            "description": "Adrenal androgen precursor; energy, well-being",
            "factors": ["Age", "Stress", "Adrenal function"]
        }
    },
    "Thyroid": {
        "TSH": {
            "unit": "mIU/L",
            "optimal": "1.0-2.0",
            "standard": "0.4-4.0",
            "concern": ">2.5 (subclinical hypothyroid?), >4.5 (hypothyroid)",
            "description": "Thyroid-stimulating hormone; pituitary control of thyroid",
            "factors": ["Iodine", "Stress", "Sleep", "Autoimmunity"]
        },
        "Free T4": {
            "unit": "ng/dL",
            "optimal": "1.2-1.5",
            "standard": "0.8-1.8",
            "concern": "Low with high TSH = hypothyroid",
            "description": "Storage thyroid hormone; converted to active T3",
            "factors": ["Iodine", "Selenium", "TSH levels"]
        },
        "Free T3": {
            "unit": "pg/mL",
            "optimal": "3.0-4.0",
            "standard": "2.3-4.2",
            "concern": "Low T3 with normal T4 = conversion issue",
            "description": "Active thyroid hormone; metabolism, energy, temperature",
            "factors": ["T4 conversion", "Selenium", "Zinc", "Stress", "Illness"]
        },
        "Reverse T3": {
            "unit": "ng/dL",
            "optimal": "10-20",
            "standard": "10-24",
            "concern": ">25 (may indicate T4 shunting away from T3)",
            "description": "Inactive T3; high levels may indicate stress/illness",
            "factors": ["Stress", "Illness", "Caloric restriction", "Inflammation"]
        },
        "TPO Antibodies": {
            "unit": "IU/mL",
            "optimal": "<20",
            "standard": "<35",
            "concern": ">35 (Hashimoto's risk)",
            "description": "Thyroid peroxidase antibodies; autoimmune marker",
            "factors": ["Autoimmunity", "Gluten", "Gut health", "Selenium"]
        },
        "Thyroglobulin Antibodies": {
            "unit": "IU/mL",
            "optimal": "<20",
            "standard": "<40",
            "concern": ">40 (autoimmune thyroid disease)",
            "description": "Another thyroid autoantibody marker",
            "factors": ["Autoimmunity", "Hashimoto's", "Graves' disease"]
        }
    },
    "Vitamins & Minerals": {
        "Vitamin D (25-OH)": {
            "unit": "ng/mL",
            "optimal": "50-80",
            "standard": "30-100",
            "concern": "<30 (deficient), >100 (potential toxicity)",
            "description": "Hormone-like vitamin; immune, bone, mood, muscle",
            "factors": ["Sun exposure", "Supplementation", "Skin color", "Latitude"]
        },
        "Vitamin B12": {
            "unit": "pg/mL",
            "optimal": "600-1000",
            "standard": "200-900",
            "concern": "<400 (suboptimal), <200 (deficient)",
            "description": "Energy, nerve function, methylation, mood",
            "factors": ["Diet (animal foods)", "Absorption (intrinsic factor)", "Metformin use"]
        },
        "Folate": {
            "unit": "ng/mL",
            "optimal": ">15",
            "standard": "3-20",
            "concern": "<3 (deficient), very high may mask B12 deficiency",
            "description": "B vitamin for methylation, cell division, mood",
            "factors": ["Diet (leafy greens)", "MTHFR status", "Alcohol"]
        },
        "Magnesium (RBC)": {
            "unit": "mg/dL",
            "optimal": "5.5-6.5",
            "standard": "4.2-6.8",
            "concern": "<5.0 (insufficiency common)",
            "description": "Intracellular magnesium; 300+ enzymatic processes",
            "factors": ["Diet", "Stress (depletes)", "Medications", "GI absorption"]
        },
        "Iron": {
            "unit": "ug/dL",
            "optimal": "60-150",
            "standard": "50-170 (M), 40-150 (F)",
            "concern": "Low = anemia risk; High = iron overload",
            "description": "Serum iron; snapshot of current iron",
            "factors": ["Diet", "Menstruation", "GI absorption", "Inflammation"]
        },
        "Zinc": {
            "unit": "ug/dL",
            "optimal": "90-120",
            "standard": "60-130",
            "concern": "<70 (insufficiency)",
            "description": "Immune function, testosterone, wound healing",
            "factors": ["Diet (oysters, meat)", "Absorption", "Copper competition"]
        }
    },
    "Kidney Function": {
        "Creatinine": {
            "unit": "mg/dL",
            "optimal": "0.8-1.2",
            "standard": "0.7-1.3 (M), 0.5-1.1 (F)",
            "concern": ">1.3 (may indicate reduced kidney function)",
            "description": "Muscle metabolism byproduct; kidney filtration marker",
            "factors": ["Muscle mass", "Hydration", "Kidney function", "Diet"]
        },
        "BUN": {
            "unit": "mg/dL",
            "optimal": "10-18",
            "standard": "7-20",
            "concern": ">20 (dehydration, kidney issues, high protein)",
            "description": "Blood urea nitrogen; protein metabolism and kidney function",
            "factors": ["Protein intake", "Hydration", "Kidney function"]
        },
        "eGFR": {
            "unit": "mL/min/1.73m2",
            "optimal": ">90",
            "standard": ">60",
            "concern": "<60 (chronic kidney disease staging)",
            "description": "Estimated glomerular filtration rate; kidney function",
            "factors": ["Age", "Creatinine", "Race", "Sex"]
        }
    },
    "Liver Function": {
        "ALT": {
            "unit": "U/L",
            "optimal": "<25",
            "standard": "7-56",
            "concern": ">40 (may indicate liver stress)",
            "description": "Alanine aminotransferase; liver enzyme",
            "factors": ["Alcohol", "Medications", "Fatty liver", "Muscle damage"]
        },
        "AST": {
            "unit": "U/L",
            "optimal": "<25",
            "standard": "10-40",
            "concern": ">40 (liver or muscle damage)",
            "description": "Aspartate aminotransferase; liver and muscle enzyme",
            "factors": ["Alcohol", "Exercise (muscle damage)", "Liver disease"]
        },
        "GGT": {
            "unit": "U/L",
            "optimal": "<20",
            "standard": "9-48",
            "concern": ">50 (alcohol, fatty liver, medications)",
            "description": "Gamma-glutamyl transferase; very sensitive to alcohol",
            "factors": ["Alcohol", "Fatty liver", "Bile duct issues"]
        },
        "Albumin": {
            "unit": "g/dL",
            "optimal": "4.2-5.0",
            "standard": "3.5-5.0",
            "concern": "<3.5 (liver dysfunction, malnutrition, inflammation)",
            "description": "Protein made by liver; nutritional and liver marker",
            "factors": ["Liver function", "Protein intake", "Inflammation"]
        }
    },
    "Blood Count": {
        "Hemoglobin": {
            "unit": "g/dL",
            "optimal": "14-16 (M), 12-14 (F)",
            "standard": "13.5-17.5 (M), 12-16 (F)",
            "concern": "Low = anemia; High = polycythemia",
            "description": "Oxygen-carrying protein in red blood cells",
            "factors": ["Iron", "B12", "Folate", "Altitude", "TRT"]
        },
        "Hematocrit": {
            "unit": "%",
            "optimal": "42-48 (M), 37-43 (F)",
            "standard": "38-50 (M), 36-44 (F)",
            "concern": ">52% (increased clot risk, especially on TRT)",
            "description": "Percentage of blood volume that is red blood cells",
            "factors": ["Hydration", "Altitude", "TRT", "EPO"]
        },
        "RBC": {
            "unit": "million/uL",
            "optimal": "4.5-5.5 (M), 4.0-5.0 (F)",
            "standard": "4.5-5.9 (M), 4.1-5.1 (F)",
            "concern": "Low = anemia; High = polycythemia",
            "description": "Red blood cell count",
            "factors": ["Iron", "EPO", "Altitude", "TRT"]
        },
        "WBC": {
            "unit": "thousand/uL",
            "optimal": "4.5-7.5",
            "standard": "4.5-11.0",
            "concern": ">11 (infection, inflammation); <4 (immune suppression)",
            "description": "White blood cell count; immune cells",
            "factors": ["Infection", "Stress", "Medications", "Autoimmunity"]
        },
        "Platelets": {
            "unit": "thousand/uL",
            "optimal": "200-300",
            "standard": "150-400",
            "concern": "<150 (bleeding risk); >400 (clotting risk)",
            "description": "Clotting cells",
            "factors": ["Bone marrow function", "Medications", "Infections"]
        }
    }
}

# =====================================================
# LANGCHAIN DOCUMENT PROCESSOR CLASS (Enhanced)
# =====================================================
class HealthDocumentProcessor:
    """Advanced document processor with LangChain integration for health documents."""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.vector_store = None
        self.processed_hashes: set = set()
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.embeddings = OpenAIEmbeddings()
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", ", ", " ", ""]
                )
                self.chat_memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            except Exception as e:
                st.warning(f"LangChain initialization warning: {e}")
                self.embeddings = None
                self.text_splitter = None
                self.chat_memory = None
        else:
            self.embeddings = None
            self.text_splitter = None
            self.chat_memory = None
    
    def get_file_hash(self, file_content: bytes) -> str:
        """Generate unique hash for file to prevent duplicate processing."""
        return hashlib.md5(file_content).hexdigest()
    
    def is_already_processed(self, file_hash: str) -> bool:
        """Check if file was already processed."""
        return file_hash in self.processed_hashes
    
    def load_document_fallback(self, file_content: bytes, file_type: str, filename: str) -> List[Document]:
        """Fallback document loading without LangChain loaders."""
        documents = []
        text_content = ""
        
        try:
            if file_type == "txt":
                text_content = file_content.decode('utf-8', errors='ignore')
            
            elif file_type == "csv":
                df = pd.read_csv(BytesIO(file_content))
                text_content = f"CSV Data from {filename}:\n\n"
                text_content += df.to_string()
            
            elif file_type == "tsv":
                df = pd.read_csv(BytesIO(file_content), sep='\t')
                text_content = f"TSV Data from {filename}:\n\n"
                text_content += df.to_string()
            
            elif file_type == "json":
                data = json.loads(file_content.decode('utf-8'))
                text_content = f"JSON Data from {filename}:\n\n"
                text_content += json.dumps(data, indent=2)
            
            elif file_type in ["yaml", "yml"]:
                try:
                    import yaml
                    data = yaml.safe_load(file_content.decode('utf-8'))
                    text_content = f"YAML Data from {filename}:\n\n"
                    text_content += json.dumps(data, indent=2)
                except ImportError:
                    text_content = file_content.decode('utf-8', errors='ignore')
            
            elif file_type == "xml":
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(file_content, 'xml')
                    text_content = soup.get_text(separator='\n', strip=True)
                else:
                    text_content = file_content.decode('utf-8', errors='ignore')
            
            elif file_type in ["html", "htm"]:
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(file_content, 'html.parser')
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text_content = soup.get_text(separator='\n', strip=True)
                else:
                    text_content = file_content.decode('utf-8', errors='ignore')
            
            elif file_type in ["md", "markdown"]:
                text_content = file_content.decode('utf-8', errors='ignore')
            
            elif file_type == "pdf" and PYPDF2_AVAILABLE:
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                text_content = f"PDF Document: {filename}\n\n"
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    text_content += f"--- Page {i+1} ---\n{page_text}\n\n"
            
            elif file_type == "docx" and DOCX_AVAILABLE:
                doc = docx.Document(BytesIO(file_content))
                text_content = f"Word Document: {filename}\n\n"
                for para in doc.paragraphs:
                    text_content += para.text + "\n"
            
            elif file_type in ["xlsx", "xls"] and OPENPYXL_AVAILABLE:
                df = pd.read_excel(BytesIO(file_content))
                text_content = f"Excel Data from {filename}:\n\n"
                text_content += df.to_string()
            
            else:
                # Try to read as text
                try:
                    text_content = file_content.decode('utf-8', errors='ignore')
                except:
                    text_content = f"Unable to extract text from {filename}"
            
            if text_content:
                documents.append(Document(
                    page_content=text_content,
                    metadata={
                        "source": filename,
                        "type": file_type,
                        "processed_at": datetime.now().isoformat(),
                        "method": "fallback_loader"
                    }
                ))
                
        except Exception as e:
            documents.append(Document(
                page_content=f"Error processing {filename}: {str(e)}",
                metadata={"source": filename, "error": str(e)}
            ))
        
        return documents
    
    def process_image_with_vision(self, image_bytes: bytes, filename: str) -> List[Document]:
        """Process image using GPT-4 Vision for OCR and analysis."""
        try:
            api_key = get_api_key()
            if not api_key:
                return [Document(
                    page_content="API key not configured. Unable to process image.",
                    metadata={"source": filename, "error": "No API key"}
                )]
            
            client = OpenAI(api_key=api_key)
            b64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine image mime type
            ext = filename.split('.')[-1].lower()
            mime_map = {
                'png': 'image/png',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'webp': 'image/webp',
                'gif': 'image/gif',
                'bmp': 'image/bmp',
                'tiff': 'image/tiff',
                'tif': 'image/tiff',
            }
            mime_type = mime_map.get(ext, 'image/jpeg')
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert medical document analyzer and OCR specialist.

YOUR TASK:
1. Extract ALL text, numbers, and data from this image with 100% accuracy
2. If it's a lab report, create a structured extraction:
   - Patient information (if visible)
   - Test date
   - Each biomarker: Name, Value, Unit, Reference Range, Flag (H/L/Normal)
3. If it's a medical document, extract all clinical information
4. If it's a prescription, extract medication names, dosages, instructions
5. If it's a medical image (X-ray, etc.), describe what you observe

FORMAT YOUR RESPONSE:
- Use clear headers and sections
- Present lab values in a consistent format
- Note any values outside reference ranges
- Include any notes or comments from the document"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Please extract and analyze all content from this health document: {filename}"},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.1  # Low temperature for accuracy
            )
            
            extracted_text = response.choices[0].message.content
            
            return [Document(
                page_content=extracted_text,
                metadata={
                    "source": filename,
                    "type": "image_ocr",
                    "processed_at": datetime.now().isoformat(),
                    "method": "gpt4_vision"
                }
            )]
            
        except Exception as e:
            return [Document(
                page_content=f"Error processing image {filename}: {str(e)}",
                metadata={"source": filename, "error": str(e)}
            )]
    
    def process_document(self, file_content: bytes, file_type: str, filename: str) -> List[Document]:
        """Process a document and return Document objects."""
        
        # Check for image types - use Vision API
        if file_type in IMAGE_EXTENSIONS:
            return self.process_image_with_vision(file_content, filename)
        
        # Use fallback loaders (more compatible with Streamlit Cloud)
        return self.load_document_fallback(file_content, file_type, filename)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into optimized chunks for retrieval."""
        if self.text_splitter and LANGCHAIN_AVAILABLE:
            return self.text_splitter.split_documents(documents)
        else:
            # Simple chunking fallback
            chunked = []
            for doc in documents:
                content = doc.page_content
                chunk_size = 1500
                overlap = 200
                
                if len(content) <= chunk_size:
                    chunked.append(doc)
                else:
                    start = 0
                    chunk_num = 0
                    while start < len(content):
                        end = start + chunk_size
                        chunk_text = content[start:end]
                        
                        chunked.append(Document(
                            page_content=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk": chunk_num
                            }
                        ))
                        
                        start = end - overlap
                        chunk_num += 1
            
            return chunked
    
    def add_documents(self, documents: List[Document], file_hash: str):
        """Add documents to the processor and vector store."""
        chunks = self.chunk_documents(documents)
        self.documents.extend(chunks)
        self.processed_hashes.add(file_hash)
        
        # Create/update vector store if LangChain available
        if LANGCHAIN_AVAILABLE and self.embeddings:
            try:
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                else:
                    self.vector_store.add_documents(chunks)
            except Exception as e:
                st.warning(f"Vector store update warning: {e}")
    
    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform semantic search across all documents."""
        if self.vector_store is not None:
            try:
                return self.vector_store.similarity_search(query, k=k)
            except:
                pass
        
        # Fallback: simple keyword search
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_docs = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            # Score by word matches
            score = sum(1 for word in query_words if word in content_lower)
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]
    
    def get_all_text(self) -> str:
        """Get all document text concatenated."""
        return "\n\n---\n\n".join([doc.page_content for doc in self.documents])
    
    def summarize_documents(self) -> str:
        """Generate a comprehensive summary of all loaded documents."""
        if not self.documents:
            return "No documents loaded to summarize."
        
        try:
            api_key = get_api_key()
            if not api_key:
                return "API key not configured."
            
            client = OpenAI(api_key=api_key)
            
            # Combine document content (limit to avoid token overflow)
            combined_text = ""
            for doc in self.documents[:15]:  # Limit documents
                combined_text += doc.page_content[:2000] + "\n\n---\n\n"
            
            if len(combined_text) > 15000:
                combined_text = combined_text[:15000] + "...[truncated]"
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a medical document analyst. Provide a comprehensive summary of the patient's health documents.

INCLUDE:
1. Document Overview: What types of documents were provided
2. Key Health Metrics: Important biomarkers and their values
3. Notable Findings: Any values outside optimal/normal ranges
4. Trends: Any patterns or changes if multiple timepoints
5. Areas of Concern: Issues that may need attention
6. Positive Findings: Health markers that look good

Format with clear headers and bullet points."""
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize these health documents:\n\n{combined_text}"
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def extract_biomarkers(self) -> Dict[str, Any]:
        """Extract and structure biomarker data from documents."""
        if not self.documents:
            return {"error": "No documents loaded"}
        
        try:
            api_key = get_api_key()
            if not api_key:
                return {"error": "API key not configured"}
            
            client = OpenAI(api_key=api_key)
            
            # Search for relevant content
            relevant_docs = self.semantic_search("lab results biomarkers blood test values complete blood count metabolic panel", k=10)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            if len(context) > 12000:
                context = context[:12000]
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a clinical data extraction specialist. Extract ALL biomarkers and lab values from the provided content.

Return ONLY valid JSON in this exact format:
{
    "extraction_date": "YYYY-MM-DD",
    "biomarkers": [
        {
            "name": "Biomarker Name",
            "value": "numeric value",
            "unit": "unit of measurement",
            "reference_range": "low-high",
            "status": "optimal|normal|elevated|low|critical",
            "category": "Metabolic|Lipid|Hormone|Thyroid|Vitamin|Inflammation|Blood Count|Other"
        }
    ],
    "document_dates": ["dates found in documents"],
    "key_findings": ["important observations"],
    "flags": ["values outside normal range"]
}

Extract every single biomarker you can find. Be thorough."""
                    },
                    {
                        "role": "user",
                        "content": f"Extract all biomarkers from this content:\n\n{context}"
                    }
                ],
                max_tokens=3000,
                temperature=0
            )
            
            response_text = response.choices[0].message.content
            
            # Try to parse JSON
            try:
                # Clean up response if needed
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                
                return json.loads(response_text.strip())
            except json.JSONDecodeError:
                return {"raw_response": response_text, "parse_error": "Could not parse as JSON"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def clear(self):
        """Clear all processed documents."""
        self.documents = []
        self.vector_store = None
        self.processed_hashes = set()
        if LANGCHAIN_AVAILABLE and self.chat_memory:
            self.chat_memory.clear()


# =====================================================
# API KEY MANAGEMENT (Streamlit Cloud Compatible)
# =====================================================
def get_api_key() -> Optional[str]:
    """Get OpenAI API key from secrets or environment."""
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except:
        pass
    
    # Try environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    
    # Try session state (user input)
    if 'user_api_key' in st.session_state and st.session_state.user_api_key:
        return st.session_state.user_api_key
    
    return None


def check_api_key() -> bool:
    """Check if API key is available and valid."""
    api_key = get_api_key()
    return api_key is not None and len(api_key) > 20


# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
def initialize_state():
    """Initialize all session state variables."""
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        
    if "messages" not in st.session_state:
        st.session_state.messages = {k: [] for k in SPECIALISTS.keys()}
    
    if "patient_data" not in st.session_state:
        # Generate realistic sample biomarker data
        np.random.seed(42)
        dates = [datetime.now() - timedelta(days=i*7) for i in range(24, 0, -1)]
        
        st.session_state.patient_data = pd.DataFrame({
            "Date": dates,
            # Metabolic
            "HbA1c": np.clip(np.random.normal(5.2, 0.15, 24).cumsum()*0.01 + 5.0, 4.5, 6.0),
            "Fasting_Glucose": np.clip(np.random.normal(88, 4, 24), 75, 105),
            "Fasting_Insulin": np.clip(np.random.normal(5, 1.5, 24), 2, 12),
            # Lipids
            "Total_Cholesterol": np.clip(np.random.normal(185, 12, 24), 150, 240),
            "LDL": np.clip(np.random.normal(105, 12, 24), 70, 160),
            "HDL": np.clip(np.random.normal(58, 6, 24), 40, 80),
            "Triglycerides": np.clip(np.random.normal(85, 18, 24), 50, 150),
            # Inflammation
            "CRP": np.clip(np.abs(np.random.normal(0.7, 0.4, 24)), 0.1, 3.0),
            "Homocysteine": np.clip(np.random.normal(8, 1.5, 24), 5, 14),
            # Hormones
            "Testosterone": np.clip(np.random.normal(680, 60, 24), 450, 900),
            "Free_T": np.clip(np.random.normal(18, 3, 24), 10, 28),
            "Estradiol": np.clip(np.random.normal(28, 6, 24), 15, 50),
            "SHBG": np.clip(np.random.normal(35, 8, 24), 18, 60),
            # Thyroid
            "TSH": np.clip(np.random.normal(1.8, 0.4, 24), 0.8, 3.5),
            "Free_T4": np.clip(np.random.normal(1.3, 0.15, 24), 0.9, 1.7),
            "Free_T3": np.clip(np.random.normal(3.2, 0.3, 24), 2.5, 4.0),
            # Vitamins
            "Vitamin_D": np.clip(np.random.normal(55, 10, 24), 25, 85),
            "Vitamin_B12": np.clip(np.random.normal(650, 100, 24), 350, 1000),
            "Ferritin": np.clip(np.random.normal(80, 25, 24), 30, 180),
            # Other
            "Body_Fat_Pct": np.clip(np.random.normal(17, 1.5, 24), 10, 28),
            "RHR": np.clip(np.random.normal(58, 4, 24), 45, 75),
            "HRV": np.clip(np.random.normal(55, 10, 24), 25, 90),
            # New Biomarkers
            "VO2_Max": np.clip(np.random.normal(48, 4, 24), 35, 65),
            "Grip_Strength": np.clip(np.random.normal(110, 10, 24), 80, 150),
            "Biological_Age": np.clip(np.random.normal(32, 2, 24), 25, 45),
            "Sleep_Score": np.clip(np.random.normal(82, 5, 24), 60, 98),
        })
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Executive Dashboard"
    
    if "active_specialist" not in st.session_state:
        st.session_state.active_specialist = list(SPECIALISTS.keys())[0]
    
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = HealthDocumentProcessor()
    
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = []
    
    if "document_summary" not in st.session_state:
        st.session_state.document_summary = None
    
    if "extracted_biomarkers" not in st.session_state:
        st.session_state.extracted_biomarkers = None
    
    if "user_api_key" not in st.session_state:
        st.session_state.user_api_key = None
    
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "name": "Premium User",
            "age": 35,
            "sex": "Male",
            "height_in": 70,
            "weight_lbs": 175,
            "activity_level": "Moderately Active",
            "goals": ["Longevity", "Energy", "Cognitive Performance"]
        }

initialize_state()


# =====================================================
# AI SPECIALIST ENGINE
# =====================================================
def call_ai_specialist(
    specialist_name: str, 
    user_input: str, 
    history: List[Dict],
    image: Optional[Any] = None
) -> str:
    """Enhanced AI specialist consultation with document context."""
    
    try:
        api_key = get_api_key()
        if not api_key:
            return """**API Key Required**

To use AI Hope Wellness specialists, please configure your OpenAI API key:

1. Go to the **Settings** page
2. Enter your OpenAI API key in the API Configuration section
3. Or set `OPENAI_API_KEY` in Streamlit secrets for deployment

Get your API key at: https://platform.openai.com/api-keys"""
        
        client = OpenAI(api_key=api_key)
        spec = SPECIALISTS[specialist_name]
        processor = st.session_state.doc_processor
        
        # Build context from uploaded documents
        doc_context = ""
        if processor.documents:
            relevant_docs = processor.semantic_search(user_input, k=5)
            if relevant_docs:
                doc_context = "\n\n--- RELEVANT PATIENT DOCUMENTS ---\n"
                for i, doc in enumerate(relevant_docs, 1):
                    # Truncate long documents
                    content = doc.page_content[:1500]
                    if len(doc.page_content) > 1500:
                        content += "...[truncated]"
                    doc_context += f"\n[Document {i} - {doc.metadata.get('source', 'Unknown')}]:\n{content}\n"
                doc_context += "\n--- END DOCUMENTS ---\n"
        
        # Build patient context
        profile = st.session_state.user_profile
        patient_context = f"""
PATIENT PROFILE:
- Age: {profile['age']} | Sex: {profile['sex']}
- Height: {profile['height_in']} inches | Weight: {profile['weight_lbs']} lbs
- Activity Level: {profile['activity_level']}
- Health Goals: {', '.join(profile['goals'])}
"""
        
        # Get latest biomarker snapshot
        latest = st.session_state.patient_data.iloc[-1]
        biomarker_snapshot = f"""
RECENT BIOMARKER SNAPSHOT:
- HbA1c: {latest['HbA1c']:.1f}% | Fasting Glucose: {latest['Fasting_Glucose']:.0f} mg/dL
- Total Testosterone: {latest['Testosterone']:.0f} ng/dL | Free T: {latest['Free_T']:.1f} pg/mL
- TSH: {latest['TSH']:.2f} mIU/L | Free T3: {latest['Free_T3']:.1f} pg/mL
- Vitamin D: {latest['Vitamin_D']:.0f} ng/mL | B12: {latest['Vitamin_B12']:.0f} pg/mL
- hs-CRP: {latest['CRP']:.2f} mg/L | Homocysteine: {latest['Homocysteine']:.1f} umol/L
- LDL: {latest['LDL']:.0f} mg/dL | HDL: {latest['HDL']:.0f} mg/dL | TG: {latest['Triglycerides']:.0f} mg/dL
- Body Fat: {latest['Body_Fat_Pct']:.1f}% | RHR: {latest['RHR']:.0f} bpm | HRV: {latest['HRV']:.0f} ms
"""
        
        system_prompt = f"""{spec['prompt']}

---

PLATFORM: {BRAND_NAME} | Created by {CREATED_BY}

{patient_context}

{biomarker_snapshot}

{doc_context}

RESPONSE GUIDELINES:
1. Be thorough but organized with clear headers
2. Reference specific biomarker values when relevant
3. Cite research when making recommendations
4. Maintain a premium, professional, supportive tone
5. Always include an appropriate medical disclaimer

Current Date: {datetime.now().strftime('%B %d, %Y')}
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 10 messages)
        for m in history[-10:]:
            messages.append({"role": m["role"], "content": m["content"]})
        
        # Handle image input
        if image is not None:
            try:
                b64_img = base64.b64encode(image.getvalue()).decode('utf-8')
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"[Analyzing attached clinical document]\n\n{user_input}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                })
            except:
                messages.append({"role": "user", "content": user_input})
        else:
            messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.4,
            max_tokens=4096
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "**Authentication Error**: Please check your OpenAI API key in Settings."
        elif "rate_limit" in error_msg.lower():
            return "**Rate Limit**: Too many requests. Please wait a moment and try again."
        elif "model" in error_msg.lower():
            return "**Model Error**: The AI model is temporarily unavailable. Please try again."
        else:
            return f"**System Error**: {error_msg}\n\nPlease check your configuration and try again."


def call_document_qa(question: str, specialist_name: str) -> Tuple[str, List[Document]]:
    """Document-based Q&A using retrieval."""
    processor = st.session_state.doc_processor
    
    if not processor.documents:
        return "No documents have been uploaded yet. Please upload health documents first.", []
    
    # Get relevant documents
    relevant_docs = processor.semantic_search(question, k=6)
    
    if not relevant_docs:
        return "No relevant information found in your documents for this question.", []
    
    # Build context
    context = "\n\n---\n\n".join([
        f"[From: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in relevant_docs
    ])
    
    # Create augmented question
    augmented_prompt = f"""Based on the patient's uploaded health documents, please answer this question:

Question: {question}

Relevant Document Excerpts:
{context}

Provide a thorough answer based on the document content. If the documents don't contain enough information, say so and provide general guidance."""
    
    # Call specialist with document context
    answer = call_ai_specialist(specialist_name, augmented_prompt, [])
    
    return answer, relevant_docs


# =====================================================
# HORIZONTAL NAVIGATION
# =====================================================
nav_options = [
    "Executive Dashboard",
    "Document Intelligence",
    "Specialist Consultation",
    "Biomarker Laboratory",
    "Wellness Protocols",
    "Health Analytics",
    "Settings"
]

# Use streamlit-antd-components for a professional horizontal menu
with st.container():
    cols = st.columns([1, 4])
    with cols[0]:
        st.markdown(f'<div class="nav-logo" style="margin-top: 10px;"><span>🏥</span> {BRAND_NAME}</div>', unsafe_allow_html=True)
    with cols[1]:
        selected_page = sac.menu(
            items=[
                sac.MenuItem("Executive Dashboard", icon="speedometer2"),
                sac.MenuItem("Document Intelligence", icon="file-earmark-medical"),
                sac.MenuItem("Specialist Consultation", icon="people"),
                sac.MenuItem("Biomarker Laboratory", icon="activity"),
                sac.MenuItem("Wellness Protocols", icon="clipboard-check"),
                sac.MenuItem("Health Analytics", icon="graph-up"),
                sac.MenuItem("Settings", icon="gear"),
            ],
            index=nav_options.index(st.session_state.current_page),
            format_func=lambda x: x,
            direction="horizontal",
            variant="light",
            key="nav_menu"
        )

if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.rerun()

# Since Streamlit doesn't support custom JS for navigation easily without components,
# we'll use a hidden radio or buttons for the actual logic, but the horizontal view is now primary.
# We'll keep a minimal sidebar for status and specialist selection.

# =====================================================
# SIDEBAR (Status & Tools)
# =====================================================
with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem 0;'>
        <h2 style='margin:0; color: {PRIMARY_COLOR};'>🏥 Status Center</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # API Status
    if check_api_key():
        st.success("API Connected", icon="✅")
    else:
        st.warning("API Key Required", icon="⚠️")
    
    st.markdown("---")
    
    # Specialist selector for relevant pages
    if st.session_state.current_page in ["Specialist Consultation", "Document Intelligence"]:
        st.markdown("### Select Specialist")
        st.session_state.active_specialist = st.selectbox(
            "Specialist",
            options=list(SPECIALISTS.keys()),
            label_visibility="collapsed"
        )
        
        spec = SPECIALISTS[st.session_state.active_specialist]
        st.markdown(f"""
        <div style='background: #1E293B; padding: 1rem; border-radius: 12px; border: 1px solid #334155; border-left: 4px solid {spec['color']}; margin-top: 0.5rem;'>
            <h4 style='margin:0; font-size: 0.9rem;'>{spec['icon']} {st.session_state.active_specialist}</h4>
            <p style='font-size: 0.7rem; color: #94A3B8; margin: 0.5rem 0 0 0; line-height: 1.4;'>{spec['bio'][:100]}...</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Document stats
    st.markdown("### Document Status")
    num_docs = len(st.session_state.doc_processor.documents)
    num_files = len(st.session_state.uploaded_files_info)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Files", num_files)
    with col2:
        st.metric("Chunks", num_docs)
    
    if st.session_state.doc_processor.vector_store is not None:
        st.success("Vector Store Active", icon="✅")
    elif num_docs > 0:
        st.info("Basic Search Active", icon="ℹ️")
    
    st.markdown("---")
    
    # Session controls
    if st.button("🔒 Clear Session", use_container_width=True):
        # Clear specific keys but keep API key
        keys_to_clear = ['messages', 'doc_processor', 'uploaded_files_info', 
                        'document_summary', 'extracted_biomarkers']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# =====================================================
# MAIN PAGE CONTENT
# =====================================================

# --- EXECUTIVE DASHBOARD ---
if st.session_state.current_page == "Executive Dashboard":
    st.markdown(f"""
    <div class="main-header">
        <span class="langchain-badge">🦜 LangChain Powered</span>
        <span class="version-badge">v{VERSION}</span>
        <h1 style='margin:0; color: {PRIMARY_COLOR}; font-size: 2.5rem;'>{BRAND_NAME}</h1>
        <h2 style='color: {SECONDARY_COLOR}; font-weight: 500; font-size: 1.25rem; margin-top: 0.75rem;'>Executive Health Intelligence Dashboard</h2>
        <p style='color: #64748B; margin-top: 1rem; font-size: 1rem;'>Welcome back. Your comprehensive health trajectory and AI-powered insights await.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key Warning
    if not check_api_key():
        st.warning("""
        **OpenAI API Key Required**
        
        To enable AI specialist consultations and document analysis, please configure your API key in Settings.
        """, icon="⚠️")
    
    # Key Metrics Row
    st.markdown("### 📈 Key Biomarkers at a Glance")
    
    latest = st.session_state.patient_data.iloc[-1]
    prev = st.session_state.patient_data.iloc[-2]
    
    metrics = [
        ("HbA1c", f"{latest['HbA1c']:.1f}%", latest['HbA1c']-prev['HbA1c'], True, "Metabolic control"),
        ("Testosterone", f"{latest['Testosterone']:.0f} ng/dL", latest['Testosterone']-prev['Testosterone'], False, "Hormonal health"),
        ("hs-CRP", f"{latest['CRP']:.2f} mg/L", latest['CRP']-prev['CRP'], True, "Inflammation"),
        ("Vitamin D", f"{latest['Vitamin_D']:.0f} ng/mL", latest['Vitamin_D']-prev['Vitamin_D'], False, "Immune function"),
        ("Body Fat", f"{latest['Body_Fat_Pct']:.1f}%", latest['Body_Fat_Pct']-prev['Body_Fat_Pct'], True, "Composition"),
        ("VO2 Max", f"{latest['VO2_Max']:.1f}", latest['VO2_Max']-prev['VO2_Max'], False, "Cardio Fitness"),
        ("Sleep Score", f"{latest['Sleep_Score']:.0f}", latest['Sleep_Score']-prev['Sleep_Score'], False, "Recovery Quality"),
        ("Bio Age", f"{latest['Biological_Age']:.1f}", latest['Biological_Age']-prev['Biological_Age'], True, "Longevity Marker"),
    ]
    
    cols = st.columns(4)
    for i, (name, value, delta, inverse, desc) in enumerate(metrics):
        with cols[i % 4]:
            st.metric(
                label=name,
                value=value,
                delta=f"{delta:+.2f}",
                delta_color="inverse" if inverse else "normal",
                help=desc
            )
    
    st.markdown("---")
    
    # Charts Row
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Hormonal & Micronutrient Trajectory")
        fig = px.line(
            st.session_state.patient_data,
            x="Date",
            y=["Testosterone", "Vitamin_D"],
            title="",
            template="plotly_white",
            color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR]
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="x unified",
            height=350
        )
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("#### Metabolic Stability (HbA1c)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=st.session_state.patient_data["Date"],
            y=st.session_state.patient_data["HbA1c"],
            fill='tozeroy',
            fillcolor=f'rgba(245, 158, 11, 0.15)',
            line=dict(color=ACCENT_COLOR, width=3),
            name='HbA1c'
        ))
        # Add optimal range band
        fig2.add_hrect(y0=4.5, y1=5.2, fillcolor="green", opacity=0.1, 
                       annotation_text="Optimal", annotation_position="top right")
        fig2.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20),
            height=350,
            showlegend=False,
            yaxis_title="HbA1c (%)"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Lipid & Thyroid Row
    st.markdown("### 🫀 Cardiovascular & Thyroid Health")
    
    lip_cols = st.columns(6)
    
    lipid_metrics = [
        ("HDL", f"{latest['HDL']:.0f}", "mg/dL", ">60 optimal"),
        ("LDL", f"{latest['LDL']:.0f}", "mg/dL", "<100 optimal"),
        ("Triglycerides", f"{latest['Triglycerides']:.0f}", "mg/dL", "<75 optimal"),
        ("TSH", f"{latest['TSH']:.2f}", "mIU/L", "1.0-2.0 optimal"),
        ("Free T3", f"{latest['Free_T3']:.1f}", "pg/mL", "3.0-4.0 optimal"),
        ("Homocysteine", f"{latest['Homocysteine']:.1f}", "umol/L", "<7 optimal"),
    ]
    
    for col, (name, value, unit, optimal) in zip(lip_cols, lipid_metrics):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="font-size: 1.8rem;">{value}</div>
                <div class="stat-label">{name}</div>
                <small style="color: #94A3B8; font-size: 0.7rem;">{unit}<br>{optimal}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Document Status & Quick Actions
    st.markdown("### 📁 Document Intelligence Status")
    
    doc_cols = st.columns([2, 1])
    
    with doc_cols[0]:
        if st.session_state.doc_processor.documents:
            st.success(f"**{len(st.session_state.uploaded_files_info)}** document(s) loaded and indexed")
            st.info(f"**{len(st.session_state.doc_processor.documents)}** searchable chunks in {'vector' if st.session_state.doc_processor.vector_store else 'keyword'} store")
            
            # Show recent files
            if st.session_state.uploaded_files_info:
                with st.expander("View uploaded files"):
                    for f in st.session_state.uploaded_files_info[-5:]:
                        st.write(f"• {f['name']} ({f['type'].upper()})")
        else:
            st.warning("No documents uploaded yet. Upload your health records in the Document Intelligence section for AI-powered analysis.")
    
    with doc_cols[1]:
        if st.button("📤 Upload Documents", use_container_width=True, type="primary"):
            st.session_state.current_page = "Document Intelligence"
            st.rerun()
        
        if st.button("💬 Consult Specialist", use_container_width=True):
            st.session_state.current_page = "Specialist Consultation"
            st.rerun()
    
    # Specialist Quick Access
    st.markdown("---")
    st.markdown("### 👥 Your Specialist Team")
    
    spec_cols = st.columns(len(SPECIALISTS))
    for col, (name, spec) in zip(spec_cols, SPECIALISTS.items()):
        with col:
            if st.button(f"{spec['icon']}\n{name.split()[0]}", use_container_width=True, help=spec['bio']):
                st.session_state.active_specialist = name
                st.session_state.current_page = "Specialist Consultation"
                st.rerun()


# --- DOCUMENT INTELLIGENCE ---
elif st.session_state.current_page == "Document Intelligence":
    st.markdown(f"""
    <div class="main-header">
        <span class="langchain-badge">🦜 LangChain Document Processing</span>
        <h1 style='margin:0; color: {PRIMARY_COLOR};'>Document Intelligence Center</h1>
        <p style='color: #64748B;'>Upload, process, and query your health documents with advanced AI analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API
    if not check_api_key():
        st.warning("**Note:** Image OCR requires an OpenAI API key. Text documents can still be processed.", icon="ℹ️")
    
    # Supported Formats Display
    with st.expander("📋 Supported File Formats (Click to expand)", expanded=False):
        for category, formats in SUPPORTED_FORMATS.items():
            st.markdown(f"**{category}**")
            format_chips = " | ".join([f"`{ext}` {info['icon']} {info['desc']}" for ext, info in formats.items()])
            st.markdown(format_chips)
        st.markdown("---")
        st.info("**Image OCR**: PNG, JPG, JPEG, WebP, GIF, BMP, TIFF images are processed using GPT-4 Vision for intelligent text extraction and medical document analysis.")
    
    # File Upload Section
    st.markdown("### 📤 Upload Health Documents")
    
    uploaded_files = st.file_uploader(
        "Drag and drop files here or click to browse",
        type=ALL_EXTENSIONS,
        accept_multiple_files=True,
        help="Upload lab reports, medical records, prescriptions, and other health documents."
    )
    
    if uploaded_files:
        st.markdown("### 📄 Processing Queue")
        
        # Show files to be processed
        new_files = []
        for file in uploaded_files:
            file_hash = st.session_state.doc_processor.get_file_hash(file.getvalue())
            if not st.session_state.doc_processor.is_already_processed(file_hash):
                new_files.append((file, file_hash))
                st.write(f"• **{file.name}** - Ready to process")
            else:
                st.write(f"• ~~{file.name}~~ - Already processed")
        
        if new_files:
            if st.button("🚀 Process New Documents", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (file, file_hash) in enumerate(new_files):
                    progress = (idx + 1) / len(new_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {file.name}")
                    
                    file_ext = file.name.split('.')[-1].lower()
                    
                    # Process document
                    docs = st.session_state.doc_processor.process_document(
                        file.getvalue(),
                        file_ext,
                        file.name
                    )
                    
                    # Add to processor
                    st.session_state.doc_processor.add_documents(docs, file_hash)
                    
                    # Track file info
                    st.session_state.uploaded_files_info.append({
                        'name': file.name,
                        'type': file_ext,
                        'hash': file_hash,
                        'chunks': len(docs),
                        'processed_at': datetime.now().isoformat()
                    })
                
                progress_bar.progress(1.0)
                status_text.text("✅ All documents processed successfully!")
                st.success(f"Processed {len(new_files)} new document(s)")
                st.rerun()
        else:
            st.info("All uploaded files have already been processed.")
    
    # Document Library
    if st.session_state.uploaded_files_info:
        st.markdown("---")
        st.markdown("### 📚 Document Library")
        
        for file_info in st.session_state.uploaded_files_info:
            ext = file_info['type']
            icon = "📄"
            for cat_formats in SUPPORTED_FORMATS.values():
                if ext in cat_formats:
                    icon = cat_formats[ext]['icon']
                    break
            
            st.markdown(f"""
            <div class="chunk-card">
                <strong>{icon} {file_info['name']}</strong>
                <br><small style="color: #64748B;">Type: {ext.upper()} | Chunks: {file_info['chunks']} | Processed: {file_info.get('processed_at', 'Unknown')[:10]}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Document Q&A
        st.markdown("---")
        st.markdown("### 🔍 Ask Questions About Your Documents")
        
        query = st.text_input(
            "Enter your question",
            placeholder="What are my vitamin D levels? Are there any concerning biomarkers? What does my thyroid panel show?"
        )
        
        if query:
            if check_api_key():
                with st.spinner("Searching documents and generating response..."):
                    answer, sources = call_document_qa(query, st.session_state.active_specialist)
                
                st.markdown("#### Response")
                st.markdown(answer)
                
                if sources:
                    with st.expander("📎 Source Documents"):
                        for i, doc in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** ({doc.metadata.get('source', 'Unknown')})")
                            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                            st.markdown("---")
            else:
                st.error("Please configure your OpenAI API key in Settings to use document Q&A.")
        
        # Summary & Extraction Tools
        st.markdown("---")
        st.markdown("### 🛠️ Analysis Tools")
        
        tool_cols = st.columns(3)
        
        with tool_cols[0]:
            if st.button("📝 Generate Summary", use_container_width=True, disabled=not check_api_key()):
                with st.spinner("Generating comprehensive summary..."):
                    summary = st.session_state.doc_processor.summarize_documents()
                    st.session_state.document_summary = summary
        
        with tool_cols[1]:
            if st.button("🧬 Extract Biomarkers", use_container_width=True, disabled=not check_api_key()):
                with st.spinner("Extracting biomarker data..."):
                    biomarkers = st.session_state.doc_processor.extract_biomarkers()
                    st.session_state.extracted_biomarkers = biomarkers
        
        with tool_cols[2]:
            if st.button("🗑️ Clear All Documents", use_container_width=True, type="secondary"):
                st.session_state.doc_processor.clear()
                st.session_state.uploaded_files_info = []
                st.session_state.document_summary = None
                st.session_state.extracted_biomarkers = None
                st.success("All documents cleared.")
                st.rerun()
        
        # Display results
        if st.session_state.document_summary:
            st.markdown("#### 📄 Document Summary")
            st.markdown(st.session_state.document_summary)
        
        if st.session_state.extracted_biomarkers:
            st.markdown("#### 🧬 Extracted Biomarkers")
            if isinstance(st.session_state.extracted_biomarkers, dict):
                if 'biomarkers' in st.session_state.extracted_biomarkers:
                    st.json(st.session_state.extracted_biomarkers)
                elif 'error' in st.session_state.extracted_biomarkers:
                    st.error(st.session_state.extracted_biomarkers['error'])
                else:
                    st.json(st.session_state.extracted_biomarkers)


# --- SPECIALIST CONSULTATION ---
elif st.session_state.current_page == "Specialist Consultation":
    spec = SPECIALISTS[st.session_state.active_specialist]
    
    st.markdown(f"""
    <div class="main-header" style="border-left-color: {spec['color']};">
        <h1 style='margin:0; color: {PRIMARY_COLOR};'>{spec['icon']} {st.session_state.active_specialist}</h1>
        <p style='color: #64748B; margin-top: 0.5rem;'>{spec['bio']}</p>
        <div style='margin-top: 1rem;'>
            {'<span class="status-badge success">📚 Document-Aware</span>' if st.session_state.doc_processor.documents else '<span class="status-badge info">No documents loaded</span>'}
            <span class="status-badge {'success' if check_api_key() else 'warning'}">{'✓ API Connected' if check_api_key() else '⚠️ API Required'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Expertise tags
    st.markdown("**Areas of Expertise:** " + " | ".join([f"`{e}`" for e in spec['expertise']]))
    
    st.markdown("---")
    
    # Check API key
    if not check_api_key():
        st.error("""
        **OpenAI API Key Required**
        
        Please configure your API key in the Settings page to enable specialist consultations.
        """)
        if st.button("Go to Settings"):
            st.session_state.current_page = "Settings"
            st.rerun()
    else:
        # Chat interface
        chat_container = st.container(height=450)
        
        with chat_container:
            for msg in st.session_state.messages[st.session_state.active_specialist]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Input area
        col1, col2 = st.columns([5, 1])
        
        with col2:
            uploaded_image = st.file_uploader(
                "📎",
                type=['png', 'jpg', 'jpeg'],
                label_visibility="collapsed",
                key="chat_image_upload",
                help="Attach an image (lab result, medical document)"
            )
        
        prompt = st.chat_input(f"Ask the {st.session_state.active_specialist}...")
        
        if prompt:
            # Add user message
            st.session_state.messages[st.session_state.active_specialist].append({
                "role": "user",
                "content": prompt
            })
            
            # Display in chat
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Consulting clinical database..."):
                        response = call_ai_specialist(
                            st.session_state.active_specialist,
                            prompt,
                            st.session_state.messages[st.session_state.active_specialist],
                            uploaded_image
                        )
                        st.markdown(response)
            
            # Save assistant message
            st.session_state.messages[st.session_state.active_specialist].append({
                "role": "assistant",
                "content": response
            })
            
            st.rerun()
        
        # Quick consultation topics
        st.markdown("---")
        st.markdown("#### 🚀 Quick Consultation Topics")
        
        quick_topics = {
            "Chief Medical Officer": [
                "Review my overall health status",
                "What should I prioritize based on my labs?",
                "Create a comprehensive health optimization plan",
                "What follow-up tests do you recommend?"
            ],
            "Peptide & Bio-Regulator Expert": [
                "BPC-157 protocol for gut healing",
                "Best peptides for injury recovery",
                "GH secretagogue stack recommendations",
                "Thymosin Alpha-1 for immune support"
            ],
            "Hormone Optimization MD": [
                "Review my testosterone levels and optimization options",
                "Thyroid panel interpretation and recommendations",
                "Signs of suboptimal hormone levels",
                "Natural vs TRT: what's right for me?"
            ],
            "Functional Nutritionist": [
                "Optimize my diet based on my biomarkers",
                "Address inflammation through nutrition",
                "Gut health protocol recommendations",
                "Key supplements I should consider"
            ],
            "Performance & Longevity Coach": [
                "How can I improve my VO2 Max?",
                "Optimal training for longevity",
                "Recovery optimization strategies",
                "Biological age reduction protocols"
            ],
            "Sleep & Circadian Specialist": [
                "Optimize my sleep quality",
                "Evening supplement protocol for sleep",
                "Fix my circadian rhythm",
                "Address sleep apnea concerns"
            ],
            "Mental Health & Neuroptimization": [
                "Cognitive enhancement stack recommendations",
                "Natural approaches to mood optimization",
                "BDNF boosting strategies",
                "Stress resilience protocol"
            ]
        }
        
        topics = quick_topics.get(st.session_state.active_specialist, [])
        
        topic_cols = st.columns(min(len(topics), 4))
        for i, topic in enumerate(topics[:4]):
            with topic_cols[i]:
                if st.button(topic, key=f"topic_{i}", use_container_width=True):
                    st.session_state.messages[st.session_state.active_specialist].append({
                        "role": "user",
                        "content": topic
                    })
                    st.rerun()
        
        # Clear conversation
        if st.session_state.messages[st.session_state.active_specialist]:
            st.markdown("---")
            if st.button("🗑️ Clear This Conversation", type="secondary"):
                st.session_state.messages[st.session_state.active_specialist] = []
                st.rerun()


# --- BIOMARKER LABORATORY ---
elif st.session_state.current_page == "Biomarker Laboratory":
    st.markdown(f"""
    <div class="main-header">
        <span class="langchain-badge">🔬 AI-Powered Analysis</span>
        <h1 style='margin:0; color: {PRIMARY_COLOR};'>Precision Biomarker Laboratory</h1>
        <p style='color: #64748B;'>Comprehensive biomarker tracking, analysis, and AI-driven interpretation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["📤 Upload & Analyze", "📊 Biomarker Tracking", "🎯 Reference Guide", "📈 Trends & Correlations"])
    
    # Tab 1: Upload & Analyze
    with tabs[0]:
        st.markdown("### Upload Lab Report for AI Analysis")
        
        lab_file = st.file_uploader(
            "Upload your lab report",
            type=['pdf', 'png', 'jpg', 'jpeg', 'docx', 'xlsx', 'csv'],
            help="Supports PDF, images, Word documents, and spreadsheets",
            key="lab_upload"
        )
        
        if lab_file:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if lab_file.type and lab_file.type.startswith('image'):
                    st.image(lab_file, caption="Lab Report Preview", use_container_width=True)
                else:
                    st.info(f"📄 **{lab_file.name}**")
                    st.write(f"Size: {lab_file.size / 1024:.1f} KB")
                    st.write(f"Type: {lab_file.type or 'Unknown'}")
            
            with col2:
                analysis_type = st.selectbox(
                    "Analysis Type",
                    [
                        "Comprehensive Analysis",
                        "Biomarker Extraction",
                        "Risk Assessment",
                        "Trend Comparison",
                        "Optimization Recommendations"
                    ]
                )
                
                specialist_for_analysis = st.selectbox(
                    "Analyzing Specialist",
                    options=list(SPECIALISTS.keys()),
                    index=0
                )
                
                if st.button("🔬 Run Analysis", type="primary", use_container_width=True, disabled=not check_api_key()):
                    with st.spinner("Analyzing lab report with AI..."):
                        # Process the document
                        file_ext = lab_file.name.split('.')[-1].lower()
                        
                        docs = st.session_state.doc_processor.process_document(
                            lab_file.getvalue(),
                            file_ext,
                            lab_file.name
                        )
                        
                        # Add to processor
                        file_hash = st.session_state.doc_processor.get_file_hash(lab_file.getvalue())
                        if not st.session_state.doc_processor.is_already_processed(file_hash):
                            st.session_state.doc_processor.add_documents(docs, file_hash)
                            st.session_state.uploaded_files_info.append({
                                'name': lab_file.name,
                                'type': file_ext,
                                'hash': file_hash,
                                'chunks': len(docs),
                                'processed_at': datetime.now().isoformat()
                            })
                        
                        # Generate analysis
                        analysis_prompt = f"""Perform a detailed {analysis_type.lower()} of this lab report.

Lab Report Content:
{docs[0].page_content if docs else 'Unable to extract content'}

Please provide:
1. **Summary**: Overview of all biomarkers found
2. **Key Findings**: Most important observations
3. **Values Outside Optimal Range**: Flag concerning values
4. **Risk Assessment**: Potential health concerns
5. **Optimization Recommendations**: Specific actions to improve markers
6. **Suggested Follow-up Tests**: Additional testing recommendations

Be thorough and specific. Reference optimal ranges (not just standard reference ranges)."""
                        
                        analysis = call_ai_specialist(
                            specialist_for_analysis,
                            analysis_prompt,
                            []
                        )
                    
                    st.markdown("### 📋 Analysis Results")
                    st.markdown(analysis)
        
        if not check_api_key():
            st.warning("Configure your OpenAI API key in Settings to enable AI analysis.")
    
    # Tab 2: Biomarker Tracking
    with tabs[1]:
        st.markdown("### Biomarker History & Trends")
        
        available_markers = [col for col in st.session_state.patient_data.columns if col != 'Date']
        
        selected_markers = st.multiselect(
            "Select biomarkers to track",
            options=available_markers,
            default=["HbA1c", "Testosterone", "CRP", "Vitamin_D"],
            help="Choose multiple biomarkers to visualize their trends over time"
        )
        
        if selected_markers:
            fig = px.line(
                st.session_state.patient_data,
                x="Date",
                y=selected_markers,
                title="Biomarker Trends Over Time",
                template="plotly_white"
            )
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                hovermode="x unified",
                height=400
            )
            fig.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics table
            st.markdown("### 📊 Statistics")
            stats_data = []
            for marker in selected_markers:
                current = st.session_state.patient_data[marker].iloc[-1]
                previous = st.session_state.patient_data[marker].iloc[-2]
                change = current - previous
                avg = st.session_state.patient_data[marker].mean()
                
                stats_data.append({
                    "Marker": marker.replace("_", " "),
                    "Current": f"{current:.2f}",
                    "Previous": f"{previous:.2f}",
                    "Change": f"{change:+.2f}",
                    "Average": f"{avg:.2f}",
                    "Min": f"{st.session_state.patient_data[marker].min():.2f}",
                    "Max": f"{st.session_state.patient_data[marker].max():.2f}",
                    "Trend": "📈" if change > 0 else "📉" if change < 0 else "➡️"
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
    
    # Tab 3: Reference Guide
    with tabs[2]:
        st.markdown("### Biomarker Reference Guide")
        st.info("**Note:** Optimal ranges shown here are for longevity optimization and may differ from standard lab reference ranges.")
        
        # Category filter
        categories = list(BIOMARKER_DATABASE.keys())
        selected_category = st.selectbox("Filter by Category", ["All Categories"] + categories)
        
        # Search
        search_term = st.text_input("Search biomarkers", placeholder="e.g., testosterone, glucose, TSH")
        
        # Display biomarkers
        for category, markers in BIOMARKER_DATABASE.items():
            if selected_category != "All Categories" and category != selected_category:
                continue
            
            with st.expander(f"**{category}**", expanded=(selected_category == category)):
                for marker_name, marker_info in markers.items():
                    # Search filter
                    if search_term and search_term.lower() not in marker_name.lower():
                        continue
                    
                    st.markdown(f"""
                    <div class="biomarker-row">
                        <div>
                            <strong>{marker_name}</strong>
                            <br><small style="color: #64748B;">{marker_info['description']}</small>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: {SECONDARY_COLOR}; font-weight: 600;">Optimal: {marker_info['optimal']}</span>
                            <br><small style="color: #94A3B8;">Standard: {marker_info['standard']} {marker_info['unit']}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Tab 4: Trends & Correlations
    with tabs[3]:
        st.markdown("### Advanced Trend Analysis")
        
        # Single marker deep dive
        trend_marker = st.selectbox(
            "Select biomarker for detailed trend analysis",
            options=[col for col in st.session_state.patient_data.columns if col != 'Date']
        )
        
        if trend_marker:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_trend = go.Figure()
                
                # Actual values
                fig_trend.add_trace(go.Scatter(
                    x=st.session_state.patient_data['Date'],
                    y=st.session_state.patient_data[trend_marker],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color=PRIMARY_COLOR, width=2.5),
                    marker=dict(size=6)
                ))
                
                # Rolling average
                rolling_avg = st.session_state.patient_data[trend_marker].rolling(window=4).mean()
                fig_trend.add_trace(go.Scatter(
                    x=st.session_state.patient_data['Date'],
                    y=rolling_avg,
                    mode='lines',
                    name='4-Week Average',
                    line=dict(color=SECONDARY_COLOR, width=2, dash='dash')
                ))
                
                fig_trend.update_layout(
                    title=f"{trend_marker.replace('_', ' ')} Trend Analysis",
                    template="plotly_white",
                    hovermode="x unified",
                    height=350
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Statistics
                data = st.session_state.patient_data[trend_marker]
                st.markdown("#### Statistics")
                st.metric("Current", f"{data.iloc[-1]:.2f}")
                st.metric("4-Week Avg", f"{data.tail(4).mean():.2f}")
                st.metric("All-Time Avg", f"{data.mean():.2f}")
                st.metric("Std Dev", f"{data.std():.2f}")


# --- WELLNESS PROTOCOLS ---
elif st.session_state.current_page == "Wellness Protocols":
    st.markdown(f"""
    <div class="main-header">
        <h1 style='margin:0; color: {PRIMARY_COLOR};'>📋 Evidence-Based Wellness Protocols</h1>
        <p style='color: #64748B;'>Comprehensive protocols backed by research, tailored for optimization.</p>
    </div>
    """, unsafe_allow_html=True)
    
    protocol_tabs = st.tabs(["🧬 Peptides", "⚖️ Hormones", "🥗 Nutrition", "⚡ Longevity", "🌙 Sleep", "🧠 Cognitive", "💪 Exercise", "🛡️ Immune & Detox"])
    
    # Peptide Protocols
    with protocol_tabs[0]:
        st.markdown("""
        ### Therapeutic Peptide Protocols
        
        > **Important**: All peptide protocols require medical supervision. Quality sourcing and proper reconstitution are critical for safety and efficacy.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Tissue Repair & Recovery Stack
            
            **BPC-157** (Body Protection Compound)
            - **Mechanism**: Upregulates growth hormone receptors, promotes angiogenesis, modulates nitric oxide
            - **Dose**: 250-500mcg twice daily
            - **Route**: Subcutaneous injection (systemic) or oral (gut-focused)
            - **Duration**: 4-8 weeks
            - **Best for**: Gut healing, tendon/ligament repair, neuroprotection
            
            **TB-500** (Thymosin Beta-4)
            - **Mechanism**: Promotes cell migration, angiogenesis, reduces inflammation
            - **Dose**: 2.5mg twice weekly (loading), 2.5mg weekly (maintenance)
            - **Route**: Subcutaneous injection
            - **Duration**: 4-6 weeks loading, then as needed
            - **Best for**: Systemic tissue repair, cardiac health, hair growth
            
            **GHK-Cu** (Copper Peptide)
            - **Mechanism**: Collagen synthesis, antioxidant, copper delivery
            - **Dose**: 1-2mg daily (injection) or topical
            - **Route**: Subcutaneous or topical
            - **Best for**: Skin health, wound healing, anti-aging
            
            **KPV**
            - **Mechanism**: Alpha-MSH derivative, anti-inflammatory
            - **Dose**: 200-400mcg daily
            - **Route**: Oral, subcutaneous, or topical
            - **Best for**: Gut inflammation, IBD, skin conditions
            """)
        
        with col2:
            st.markdown("""
            #### Growth Hormone Optimization
            
            **CJC-1295 + Ipamorelin** (Combo)
            - **Mechanism**: GHRH analog + ghrelin mimetic for synergistic GH release
            - **Dose**: CJC 100mcg + Ipamorelin 200mcg
            - **Timing**: Before bed, fasted (no carbs 2hrs prior)
            - **Frequency**: 5 days on, 2 days off
            - **Duration**: 8-12 weeks
            
            **Tesamorelin**
            - **Mechanism**: GHRH analog, FDA-approved for lipodystrophy
            - **Dose**: 2mg daily
            - **Timing**: Before bed
            - **Best for**: Visceral fat reduction, cognitive support
            
            #### Immune Modulation
            
            **Thymosin Alpha-1**
            - **Mechanism**: Immune modulator, enhances T-cell function
            - **Dose**: 1.5mg twice weekly (prevention) or daily (acute)
            - **Route**: Subcutaneous injection
            - **Best for**: Immune enhancement, chronic infections, cancer support
            
            **LL-37**
            - **Mechanism**: Cathelicidin antimicrobial peptide
            - **Dose**: 100-200mcg daily
            - **Best for**: Antimicrobial, wound healing, biofilm disruption
            """)
        
        st.warning("""
        **⚠️ Safety Notice**: Peptides are research compounds. Many are not FDA-approved for human use. 
        Always work with a knowledgeable physician, use quality-tested sources, and monitor for adverse effects.
        """)
    
    # Hormone Protocols
    with protocol_tabs[1]:
        st.markdown("### Hormone Optimization Strategies")
        
        # Current levels display
        latest = st.session_state.patient_data.iloc[-1]
        
        st.markdown(f"""
        #### Your Current Hormone Status
        Based on latest data:
        - **Total Testosterone**: {latest['Testosterone']:.0f} ng/dL
        - **Free Testosterone**: {latest['Free_T']:.1f} pg/mL
        - **Estradiol**: {latest['Estradiol']:.0f} pg/mL
        - **SHBG**: {latest['SHBG']:.0f} nmol/L
        - **TSH**: {latest['TSH']:.2f} mIU/L
        - **Free T3**: {latest['Free_T3']:.1f} pg/mL
        """)
        
        # Assessment
        t_status = "optimal" if latest['Testosterone'] >= 700 else "suboptimal" if latest['Testosterone'] >= 500 else "low"
        
        if t_status == "optimal":
            st.success("✓ Testosterone levels are in the optimal range. Focus on maintenance strategies.")
        elif t_status == "suboptimal":
            st.warning("⚠️ Testosterone is functional but could be optimized. Consider natural support first.")
        else:
            st.error("⚠️ Testosterone is below optimal. Evaluation for underlying causes and potential TRT warranted.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Natural Testosterone Support
            
            **Foundational Supplements**
            - **Zinc**: 30mg daily (zinc citrate or picolinate)
            - **Vitamin D3**: 5000-10000 IU daily (target 50-80 ng/mL)
            - **Magnesium**: 400-600mg daily (glycinate preferred)
            - **Boron**: 6-9mg daily
            
            **Herbal Support**
            - **Tongkat Ali**: 400mg daily (10% eurycomanone)
            - **Ashwagandha**: 600mg KSM-66 daily
            - **Fadogia Agrestis**: 600mg daily
            - **Shilajit**: 500mg daily
            
            **DHEA** (if levels are low)
            - Dose: 25-50mg daily
            - Monitor: DHEA-S, testosterone, estrogen
            
            **Lifestyle Factors**
            - Strength training 3-4x/week
            - Sleep 7-9 hours (critical for T production)
            - Manage stress (cortisol suppresses T)
            - Maintain body fat 12-18%
            - Limit alcohol (increases aromatase)
            - Avoid endocrine disruptors
            """)
        
        with col2:
            st.markdown("""
            #### TRT Considerations
            
            **When to Consider TRT**
            - Total T consistently < 400 ng/dL
            - Symptoms despite optimization efforts
            - Low T with elevated LH (primary hypogonadism)
            
            **Common TRT Protocols**
            
            *Standard Protocol*
            - Testosterone Cypionate: 100-200mg weekly
            - Split into 2 injections (e.g., Mon/Thu)
            - HCG: 500 IU 2-3x weekly (fertility preservation)
            - AI only if needed (target E2: 20-35 pg/mL)
            
            *Monitoring Schedule*
            - 6 weeks: Initial labs
            - 3 months: Optimization labs
            - Every 6 months: Maintenance labs
            
            **Key Markers to Monitor**
            - Total & Free Testosterone
            - Estradiol (sensitive assay)
            - Hematocrit (watch for elevation >52%)
            - PSA (prostate screening)
            - Lipid panel
            
            #### Thyroid Optimization
            
            If TSH > 2.5 or Free T3 < 3.0:
            - Check TPO and TG antibodies
            - Optimize iodine, selenium, zinc
            - Consider T4/T3 combination therapy
            """)
        
        st.info("**Note**: Hormone optimization should be guided by comprehensive lab testing and an experienced physician.")
    
    # Nutrition Protocols
    with protocol_tabs[2]:
        st.markdown("### Metabolic Nutrition Protocols")
        
        latest = st.session_state.patient_data.iloc[-1]
        
        st.markdown(f"""
        #### Metabolic Status
        Based on your HbA1c of **{latest['HbA1c']:.1f}%** and Fasting Glucose of **{latest['Fasting_Glucose']:.0f} mg/dL**, 
        your metabolic health appears {'optimal' if latest['HbA1c'] < 5.3 else 'good with room for optimization' if latest['HbA1c'] < 5.7 else 'needing attention'}.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### Protein Protocol
            
            **Target**: 1.0-1.2g per lb lean body mass
            
            **Timing**
            - 30-50g per meal
            - 4-5 meals/day
            - Post-workout: 40-50g + leucine
            
            **Quality Sources**
            - Grass-fed beef
            - Wild-caught fish
            - Pasture-raised eggs
            - Organic poultry
            - Whey/collagen protein
            
            **Key Amino Acids**
            - Leucine: 2.5-3g per meal (muscle protein synthesis trigger)
            - Glycine: 3-5g daily (collagen, sleep)
            - Taurine: 2-3g daily (heart, longevity)
            """)
        
        with col2:
            st.markdown("""
            #### Carbohydrate Strategy
            
            **Approach**: Strategic carb cycling
            
            **Training Days**
            - Total: 100-150g
            - Peri-workout: 30-50g
            - Evening: 30-50g (for sleep)
            
            **Rest Days**
            - Total: 50-75g
            - Focus on vegetables
            - Limit starches
            
            **Optimal Sources**
            - Root vegetables
            - Berries
            - Leafy greens
            - White rice (post-workout)
            
            **Avoid**
            - Refined sugars
            - Processed grains
            - Fruit juice
            - High-fructose foods
            """)
        
        with col3:
            st.markdown("""
            #### Fats & Oils
            
            **Target**: 30-40% of calories
            
            **Prioritize**
            - Extra virgin olive oil
            - Avocado & avocado oil
            - Wild salmon (omega-3)
            - Grass-fed butter/ghee
            - Macadamia nuts
            
            **Omega-3 Target**
            - EPA + DHA: 3-4g daily
            - Ratio: Omega-6:3 < 4:1
            
            **Avoid/Minimize**
            - Seed oils (canola, soy, corn)
            - Trans fats
            - Fried foods
            - Processed snacks
            
            **MCT Oil**
            - 1-2 tbsp daily
            - Ketone production
            - Cognitive support
            """)
        
        st.markdown("---")
        st.markdown("### Key Supplement Protocol")
        
        supp_col1, supp_col2 = st.columns(2)
        
        with supp_col1:
            st.markdown("""
            #### Foundational
            - **Vitamin D3/K2**: 5000 IU / 200mcg daily
            - **Omega-3**: 3-4g EPA+DHA daily
            - **Magnesium Glycinate**: 400-600mg before bed
            - **Creatine Monohydrate**: 5g daily
            - **Collagen Peptides**: 10-20g daily
            
            #### Metabolic Support
            - **Berberine**: 500mg before largest meal
            - **Alpha Lipoic Acid**: 600mg daily
            - **Chromium**: 200mcg with meals
            """)
        
        with supp_col2:
            st.markdown("""
            #### Gut Health
            - **Probiotic**: Multi-strain, 50+ billion CFU
            - **Prebiotic Fiber**: Acacia, PHGG
            - **L-Glutamine**: 5-10g daily
            - **Digestive Enzymes**: With meals
            
            #### Anti-Inflammatory
            - **Curcumin**: 500mg (with piperine)
            - **Quercetin**: 500mg daily
            - **Ginger**: 500mg daily
            """)
    
    # Longevity Protocols
    with protocol_tabs[3]:
        st.markdown("### Evidence-Based Longevity Stack")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### NAD+ & Cellular Energy
            
            **NMN (Nicotinamide Mononucleotide)**
            - Dose: 500-1000mg daily
            - Timing: Morning (sublingual preferred)
            - Mechanism: NAD+ precursor, sirtuin activation
            - Evidence: Strong preclinical, human trials ongoing
            
            **NR (Nicotinamide Riboside)**
            - Alternative to NMN
            - Dose: 300-600mg daily
            - More established human data
            
            **Resveratrol**
            - Dose: 500mg with fat (morning)
            - Mechanism: Sirtuin activation, AMPK
            - Synergistic with NMN
            
            #### Senolytic Protocol (Monthly)
            
            **Quercetin + Fisetin Stack**
            - Quercetin: 1000mg
            - Fisetin: 500-1000mg
            - Protocol: 2 consecutive days monthly
            - Mechanism: Clears senescent cells
            
            **Autophagy Support**
            - Spermidine: 1-2mg daily
            - Extended fasting: 24-72hr quarterly
            - Exercise: Induces autophagy
            """)
        
        with col2:
            st.markdown("""
            #### Mitochondrial Support
            
            **CoQ10 (Ubiquinol)**
            - Dose: 200-300mg daily
            - Critical for energy production
            - Depleted by statins
            
            **PQQ (Pyrroloquinoline quinone)**
            - Dose: 20mg daily
            - Mitochondrial biogenesis
            
            **Acetyl L-Carnitine**
            - Dose: 1500mg daily
            - Fat transport into mitochondria
            - Cognitive support
            
            #### Hormetic Stressors
            
            **Sauna Protocol**
            - 4x per week
            - 20 minutes at 170-190°F
            - Benefits: Heat shock proteins, cardiovascular
            
            **Cold Exposure**
            - 2-3 min cold shower daily
            - OR 11 min/week total cold immersion
            - Benefits: Brown fat, dopamine, resilience
            
            **Fasting Windows**
            - Daily: 16:8 minimum
            - Weekly: 24-hour fast optional
            - Benefits: Autophagy, insulin sensitivity
            """)
        
        st.markdown("---")
        st.markdown("#### Cardiovascular Longevity Support")
        
        cv_col1, cv_col2, cv_col3 = st.columns(3)
        
        with cv_col1:
            st.markdown("""
            **Taurine**
            - Dose: 2-3g daily
            - Heart function, longevity
            """)
        
        with cv_col2:
            st.markdown("""
            **Glycine**
            - Dose: 3-5g before bed
            - Collagen, sleep, longevity
            """)
        
        with cv_col3:
            st.markdown("""
            **Citrus Bergamot**
            - Dose: 500mg twice daily
            - Lipid optimization
            """)
    
    # Sleep Protocols
    with protocol_tabs[4]:
        st.markdown("### Sleep Architecture Optimization")
        
        st.markdown("""
        #### Sleep Quality Targets
        - **Total Sleep**: 7-9 hours
        - **Deep Sleep (N3)**: 1.5-2 hours (20-25% of total)
        - **REM Sleep**: 1.5-2 hours (20-25% of total)
        - **Sleep Efficiency**: >85%
        - **Sleep Onset**: <20 minutes
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Evening Protocol (3-Hour Runway)
            
            **3 Hours Before Bed**
            - Last meal (no eating after)
            - Dim overhead lights
            - Blue light blocking glasses on
            - No intense exercise
            - Begin temperature cool-down
            
            **2 Hours Before Bed**
            - No screens or minimal (dark mode)
            - Light reading or relaxation
            - Warm bath/shower (paradoxical cooling)
            - Light stretching or yoga nidra
            
            **1 Hour Before Bed**
            
            *Supplement Stack*
            - Magnesium L-Threonate: 144mg (or Glycinate 400mg)
            - L-Theanine: 200mg
            - Glycine: 3g
            - Apigenin: 50mg (optional)
            - Tart Cherry: 500mg (natural melatonin)
            
            **At Bedtime**
            - Melatonin: 0.3-0.5mg sublingual (only if needed)
            - Room temp: 65-68°F (18-20°C)
            - Complete darkness (blackout curtains, cover LEDs)
            - White noise if needed
            """)
        
        with col2:
            st.markdown("""
            #### Morning Protocol
            
            **Immediately Upon Waking**
            - No snooze button
            - Bright light exposure within 30 min
            - Ideally outdoor sunlight for 10+ min
            
            **Within First Hour**
            - Hydration: 16-20oz water
            - Movement: Light walk or stretching
            - Delay caffeine: 90-120 min after waking
            
            **Caffeine Strategy**
            - Last caffeine: 10+ hours before bed
            - For most people: None after 2pm
            - Consider adenosine receptor sensitivity
            
            #### Advanced Sleep Hacks
            
            **Temperature Manipulation**
            - Cool bedroom (65-68°F)
            - Warm feet (socks or heating pad)
            - ChiliPad/Eight Sleep for precise control
            
            **Tracking & Optimization**
            - Oura Ring, WHOOP, or similar
            - Track sleep stages, HRV, temperature
            - Correlate with supplements/behaviors
            
            **Sleep Apnea Screening**
            If: Snoring, daytime fatigue, waking unrested
            - Home sleep test or polysomnography
            - Consider: Mouth tape, positional therapy, CPAP
            """)
        
        st.markdown("---")
        st.markdown("#### Circadian Rhythm Optimization")
        
        st.markdown("""
        | Time | Light | Meals | Activity |
        |------|-------|-------|----------|
        | 6-8 AM | Bright light (sun or 10,000 lux) | Optional coffee | Light movement |
        | 8-12 PM | Natural daylight | Protein-rich breakfast | Peak cognitive work |
        | 12-2 PM | Continue daylight | Largest meal | |
        | 2-6 PM | Natural light | Lighter snack | Exercise window |
        | 6-9 PM | Dim, warm lights | Light dinner (3hr before bed) | Wind down |
        | 9 PM+ | Near darkness | No food | Sleep prep |
        """)
    
    # Cognitive Protocols
    with protocol_tabs[5]:
        st.markdown("### Neuroptimization & Cognitive Enhancement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Daily Foundation Stack
            
            **Structural Support**
            - **Omega-3 DHA**: 2g daily (neuronal membranes)
            - **Phosphatidylserine**: 300mg daily (cell signaling)
            - **Lion's Mane**: 1000mg daily (NGF/BDNF support)
            
            **Cholinergic Support**
            - **Alpha-GPC**: 300-600mg (acetylcholine precursor)
            - **CDP-Choline**: 250-500mg (alternative to Alpha-GPC)
            
            **Memory & Learning**
            - **Bacopa Monnieri**: 300mg daily (standardized)
            - Note: Takes 8-12 weeks for full effect
            
            #### Acute Performance Stack
            
            *For focused work sessions*
            - Caffeine: 100mg
            - L-Theanine: 200mg
            - Alpha-GPC: 300mg
            - Optional: Tyrosine 500mg (if sleep-deprived)
            
            *For creative work*
            - L-Theanine: 400mg (without caffeine)
            - Lion's Mane: Extra 500mg
            - Rhodiola: 200mg
            """)
        
        with col2:
            st.markdown("""
            #### Neuroprotection
            
            **Anti-inflammatory**
            - **Curcumin (Longvida)**: 400mg daily
            - **Blueberry Extract**: 500mg daily (anthocyanins)
            - **Cocoa Flavanols**: 500mg daily (blood flow)
            
            **Antioxidant**
            - **N-Acetyl Cysteine (NAC)**: 600mg daily
            - **Alpha Lipoic Acid**: 300mg daily
            
            #### BDNF Optimization
            
            *Non-supplement interventions*
            - High-intensity exercise
            - Intermittent fasting
            - Cold exposure
            - Novel learning
            - Social connection
            - Quality sleep
            - Sunlight exposure
            
            #### Stress Resilience
            
            **Adaptogens**
            - **Ashwagandha KSM-66**: 600mg daily
            - **Rhodiola Rosea**: 200-400mg morning
            - **Reishi Mushroom**: 1g daily
            
            **Calming**
            - **L-Theanine**: 200-400mg as needed
            - **Magnolia Bark**: 200mg (cortisol)
            - **Lemon Balm**: 300mg (GABA support)
            """)
        
        st.info("**Note**: Start with one supplement at a time to assess individual response. Some nootropics require consistent use for weeks before benefits manifest.")
    
    # Immune & Detox Protocols
    with protocol_tabs[7]:
        st.markdown("### Immune Resilience & Detoxification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Immune Resilience Stack
            
            **Foundational Support**
            - **Vitamin C**: 1000mg (liposomal preferred)
            - **Zinc**: 30mg (picolinate or citrate)
            - **Quercetin**: 500mg (ionophore for zinc)
            - **Vitamin D3**: 5000-10,000 IU (target 50-80 ng/mL)
            
            **Acute Support (at first sign of illness)**
            - **Elderberry**: 500mg 3x daily
            - **Echinacea**: 400mg 3x daily
            - **High-dose Vitamin C**: 1000mg every 2-3 hours
            - **Oil of Oregano**: 200mg twice daily
            
            **Lifestyle Immune Support**
            - **Sleep**: 7-9 hours (non-negotiable)
            - **Stress**: Cortisol suppresses immune function
            - **Gut Health**: 70% of immune system is in the gut
            """)
            
        with col2:
            st.markdown("""
            #### Detoxification Support
            
            **Phase I & II Liver Support**
            - **NAC (N-Acetyl Cysteine)**: 600-1200mg daily (glutathione precursor)
            - **Milk Thistle (Silymarin)**: 300mg daily
            - **Sulforaphane**: 50mg (from broccoli sprouts)
            - **Calcium D-Glucarate**: 500mg (estrogen detox)
            
            **Binder Protocol**
            - **Activated Charcoal**: 500mg (away from food/supps)
            - **Modified Citrus Pectin**: 5g daily
            - **Chlorella**: 2-3g daily
            
            **Hormetic Detox**
            - **Sauna**: 20-30 min 4x/week (sweating out heavy metals)
            - **Hydration**: 3-4L water with electrolytes
            - **Fiber**: 35g+ daily to prevent reabsorption
            """)
            
        st.success("✓ Detoxification is a continuous process. Focus on supporting your body's natural pathways daily.")

    # Exercise Protocols
    with protocol_tabs[6]:
        st.markdown("### Exercise for Longevity & Performance")
        
        st.markdown("""
        #### The Longevity Exercise Framework
        
        Based on the "Centenarian Decathlon" concept - what physical capabilities do you want to maintain into your 80s and 90s?
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Cardiorespiratory Fitness
            
            **Zone 2 Training** (Most Important)
            - Duration: 150-180 min/week minimum
            - Intensity: Can hold conversation, nose breathing
            - Heart Rate: ~60-70% max HR
            - Benefits: Mitochondrial efficiency, metabolic flexibility
            - Examples: Walking, easy cycling, light jogging
            
            **VO2 Max Training** (Weekly)
            - 1-2 sessions per week
            - 4x4 minute intervals at 85-95% max HR
            - 3-4 min recovery between intervals
            - Benefits: Peak aerobic capacity, mortality reduction
            
            **Target VO2 Max by Age**
            - 40s: >45 mL/kg/min
            - 50s: >40 mL/kg/min
            - 60s: >35 mL/kg/min
            - 70s: >30 mL/kg/min
            """)
        
        with col2:
            st.markdown("""
            #### Strength Training
            
            **Frequency**: 3-4 sessions/week
            
            **Focus Areas for Longevity**
            - Grip strength (mortality predictor)
            - Hip hinge (deadlift patterns)
            - Squat patterns (functional independence)
            - Pulling movements (posture, back health)
            - Carrying (farmer's walks)
            
            **Rep Ranges**
            - Strength: 3-5 reps, heavy
            - Hypertrophy: 8-12 reps
            - Muscular endurance: 15-20 reps
            - Rotate through all ranges
            
            **Key Exercises**
            - Deadlift/Hip hinge
            - Squat variation
            - Rows/Pull-ups
            - Press (horizontal & vertical)
            - Carries (farmer's walk, suitcase)
            - Single-leg work (lunges, step-ups)
            """)
        
        st.markdown("---")
        
        st.markdown("""
        #### Weekly Template
        
        | Day | Focus | Duration | Notes |
        |-----|-------|----------|-------|
        | Monday | Strength (Lower) | 45-60 min | Squat, hip hinge, carries |
        | Tuesday | Zone 2 Cardio | 45-60 min | Walking, cycling, swimming |
        | Wednesday | Strength (Upper) | 45-60 min | Push, pull, carries |
        | Thursday | Zone 2 Cardio | 45-60 min | |
        | Friday | Strength (Full Body) | 45-60 min | Compound movements |
        | Saturday | VO2 Max / HIIT | 30-40 min | 4x4 intervals |
        | Sunday | Active Recovery | 30-60 min | Walk, mobility, yoga |
        """)
        
        st.markdown("""
        #### Stability & Mobility
        
        - Daily: 10-15 min mobility routine
        - Balance work: Single-leg stands, eyes closed
        - Core stability: Pallof press, dead bugs, bird dogs
        - Hip mobility: 90/90 stretches, hip CARs
        - Thoracic mobility: Foam rolling, cat-cow
        """)


# --- HEALTH ANALYTICS ---
elif st.session_state.current_page == "Health Analytics":
    st.markdown(f"""
    <div class="main-header">
        <h1 style='margin:0; color: {PRIMARY_COLOR};'>📊 Advanced Health Analytics</h1>
        <p style='color: #64748B;'>Deep insights into your health data with visualizations and correlations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overall Health Score
    st.markdown("### Composite Health Score")
    
    latest = st.session_state.patient_data.iloc[-1]
    
    # Calculate dimension scores
    scores = {
        "Metabolic": min(100, max(0, 100 - abs(latest['HbA1c'] - 5.0) * 25)),
        "Hormonal": min(100, max(0, (latest['Testosterone'] - 400) / 6)),
        "Inflammatory": min(100, max(0, 100 - latest['CRP'] * 35)),
        "Cardiovascular": min(100, max(0, 100 - abs(latest['LDL'] - 90) * 0.4 + (latest['HDL'] - 40) * 0.5)),
        "Thyroid": min(100, max(0, 100 - abs(latest['TSH'] - 1.5) * 20)),
        "Nutritional": min(100, max(0, (latest['Vitamin_D'] - 20) * 1.5 + (latest['Vitamin_B12'] - 300) * 0.05)),
    }
    
    overall_score = np.mean(list(scores.values()))
    
    score_cols = st.columns([1, 2])
    
    with score_cols[0]:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Health Score", 'font': {'size': 20}},
            delta={'reference': 75, 'increasing': {'color': SECONDARY_COLOR}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': SECONDARY_COLOR, 'thickness': 0.75},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 50], 'color': "#FEE2E2"},
                    {'range': [50, 70], 'color': "#FEF3C7"},
                    {'range': [70, 85], 'color': "#D1FAE5"},
                    {'range': [85, 100], 'color': "#A7F3D0"}
                ],
                'threshold': {
                    'line': {'color': PRIMARY_COLOR, 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=30, r=30, t=50, b=20),
            font={'family': 'Inter'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Score interpretation
        if overall_score >= 85:
            st.success("**Excellent** - Your health metrics are well-optimized")
        elif overall_score >= 70:
            st.info("**Good** - Solid foundation with room for optimization")
        elif overall_score >= 50:
            st.warning("**Moderate** - Several areas need attention")
        else:
            st.error("**Needs Work** - Multiple health metrics require intervention")
    
    with score_cols[1]:
        # Radar chart
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=f'rgba(16, 185, 129, 0.25)',
            line=dict(color=SECONDARY_COLOR, width=2.5)
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10),
                    gridcolor='#E2E8F0'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, family='Inter')
                )
            ),
            showlegend=False,
            title=dict(text="Health Dimension Scores", font=dict(size=16)),
            height=300,
            margin=dict(l=60, r=60, t=60, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Dimension breakdown
    st.markdown("#### Dimension Breakdown")
    dim_cols = st.columns(6)
    for col, (dim, score) in zip(dim_cols, scores.items()):
        with col:
            color = SECONDARY_COLOR if score >= 70 else ACCENT_COLOR if score >= 50 else DANGER_COLOR
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; border: 1px solid #E2E8F0;">
                <div style="font-size: 1.75rem; font-weight: 800; color: {color};">{score:.0f}</div>
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 600;">{dim}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Correlation Analysis
    st.markdown("### Biomarker Correlations")
    
    # Select subset for correlation
    corr_markers = st.multiselect(
        "Select biomarkers for correlation analysis",
        options=[col for col in st.session_state.patient_data.columns if col != 'Date'],
        default=["HbA1c", "Testosterone", "CRP", "Vitamin_D", "HDL", "LDL", "TSH", "Body_Fat_Pct"]
    )
    
    if len(corr_markers) >= 2:
        corr_data = st.session_state.patient_data[corr_markers]
        corr_matrix = corr_data.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect='auto'
        )
        fig_corr.update_layout(
            title="Biomarker Correlation Matrix",
            height=500,
            font={'family': 'Inter'}
        )
        fig_corr.update_traces(
            textfont=dict(size=10)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Highlight strongest correlations
        st.markdown("#### Strongest Correlations")
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    'Pair': f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}",
                    'Correlation': corr_matrix.iloc[i, j],
                    'Strength': 'Strong' if abs(corr_matrix.iloc[i, j]) > 0.7 else 'Moderate' if abs(corr_matrix.iloc[i, j]) > 0.4 else 'Weak'
                })
        
        corr_df = pd.DataFrame(correlations)
        corr_df['Abs_Corr'] = abs(corr_df['Correlation'])
        corr_df = corr_df.sort_values('Abs_Corr', ascending=False).head(10)
        corr_df['Correlation'] = corr_df['Correlation'].apply(lambda x: f"{x:+.3f}")
        
        st.dataframe(
            corr_df[['Pair', 'Correlation', 'Strength']],
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Time series decomposition
    st.markdown("### Trend Decomposition")
    
    decomp_marker = st.selectbox(
        "Select biomarker for trend analysis",
        options=[col for col in st.session_state.patient_data.columns if col != 'Date']
    )
    
    if decomp_marker:
        data = st.session_state.patient_data[decomp_marker]
        dates = st.session_state.patient_data['Date']
        
        # Create subplot
        fig = go.Figure()
        
        # Raw data
        fig.add_trace(go.Scatter(
            x=dates, y=data,
            mode='lines+markers',
            name='Actual',
            line=dict(color=PRIMARY_COLOR, width=2),
            marker=dict(size=5)
        ))
        
        # Rolling averages
        for window, color in [(4, SECONDARY_COLOR), (8, ACCENT_COLOR)]:
            rolling = data.rolling(window=window).mean()
            fig.add_trace(go.Scatter(
                x=dates, y=rolling,
                mode='lines',
                name=f'{window}-Week Avg',
                line=dict(color=color, width=2, dash='dash')
            ))
        
        # Standard deviation bands
        rolling_std = data.rolling(window=4).std()
        rolling_mean = data.rolling(window=4).mean()
        
        fig.add_trace(go.Scatter(
            x=dates, y=rolling_mean + rolling_std,
            mode='lines',
            name='Upper Band',
            line=dict(color='#94A3B8', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=rolling_mean - rolling_std,
            mode='lines',
            name='Lower Band',
            fill='tonexty',
            fillcolor='rgba(148, 163, 184, 0.1)',
            line=dict(color='#94A3B8', width=1),
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"{decomp_marker.replace('_', ' ')} - Trend Analysis with Confidence Bands",
            template="plotly_white",
            hovermode="x unified",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)


# --- SETTINGS ---
elif st.session_state.current_page == "Settings":
    st.markdown(f"""
    <div class="main-header">
        <h1 style='margin:0; color: {PRIMARY_COLOR};'>⚙️ Settings & Configuration</h1>
        <p style='color: #64748B;'>Manage your profile, API configuration, and preferences.</p>
    </div>
    """, unsafe_allow_html=True)
    
    settings_tabs = st.tabs(["👤 Profile", "🔑 API Configuration", "📁 Data Management", "ℹ️ About"])
    
    # Profile Tab
    with settings_tabs[0]:
        st.markdown("### User Profile")
        
        profile = st.session_state.user_profile
        
        col1, col2 = st.columns(2)
        
        with col1:
            profile['name'] = st.text_input("Name", value=profile['name'])
            profile['age'] = st.number_input("Age", value=profile['age'], min_value=18, max_value=120)
            profile['sex'] = st.selectbox("Biological Sex", ["Male", "Female"], index=0 if profile['sex'] == "Male" else 1)
        
        with col2:
            profile['height_in'] = st.number_input("Height (inches)", value=profile['height_in'], min_value=48, max_value=96)
            profile['weight_lbs'] = st.number_input("Weight (lbs)", value=profile['weight_lbs'], min_value=80, max_value=500)
            profile['activity_level'] = st.selectbox(
                "Activity Level",
                ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Athlete"],
                index=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Athlete"].index(profile['activity_level'])
            )
        
        st.markdown("### Health Goals")
        profile['goals'] = st.multiselect(
            "Select your primary health goals",
            ["Longevity", "Body Composition", "Energy", "Cognitive Performance", "Sleep Quality", 
             "Hormone Optimization", "Disease Prevention", "Athletic Performance", "Stress Management",
             "Gut Health", "Immune Support", "Mental Clarity"],
            default=profile['goals']
        )
        
        st.session_state.user_profile = profile
        
        if st.button("💾 Save Profile", type="primary"):
            st.success("Profile saved successfully!")
    
    # API Configuration Tab
    with settings_tabs[1]:
        st.markdown("### OpenAI API Configuration")
        
        st.info("""
        **For Streamlit Community Cloud Deployment:**
        
        Add your API key to Streamlit Secrets:
        1. Go to your app's dashboard on Streamlit Cloud
        2. Click "Settings" → "Secrets"
        3. Add: `OPENAI_API_KEY = "your-api-key-here"`
        
        **For Local Development:**
        - Set the `OPENAI_API_KEY` environment variable, OR
        - Enter it below (stored only for this session)
        """)
        
        # Current status
        if check_api_key():
            st.success("✓ OpenAI API key is configured and active", icon="✅")
            
            # Show key source
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                st.info("Key source: Streamlit Secrets")
            elif os.getenv('OPENAI_API_KEY'):
                st.info("Key source: Environment Variable")
            else:
                st.info("Key source: Session Input")
        else:
            st.warning("⚠️ OpenAI API key not configured", icon="⚠️")
            
            # Manual input option
            st.markdown("#### Enter API Key Manually")
            api_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Your API key will only be stored for this session"
            )
            
            if st.button("Save API Key"):
                if api_input and len(api_input) > 20:
                    st.session_state.user_api_key = api_input
                    st.success("API key saved for this session!")
                    st.rerun()
                else:
                    st.error("Please enter a valid API key")
        
        st.markdown("---")
        st.markdown("#### Get an API Key")
        st.markdown("""
        1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
        2. Sign in or create an account
        3. Click "Create new secret key"
        4. Copy and save your key securely
        """)
    
    # Data Management Tab
    with settings_tabs[2]:
        st.markdown("### Document Management")
        
        if st.session_state.uploaded_files_info:
            st.markdown(f"**{len(st.session_state.uploaded_files_info)}** document(s) in library")
            st.markdown(f"**{len(st.session_state.doc_processor.documents)}** searchable chunks")
            
            with st.expander("View uploaded files"):
                for f in st.session_state.uploaded_files_info:
                    st.write(f"• {f['name']} ({f['type'].upper()}) - {f['chunks']} chunks")
            
            if st.button("🗑️ Clear All Documents", type="secondary"):
                st.session_state.doc_processor.clear()
                st.session_state.uploaded_files_info = []
                st.session_state.document_summary = None
                st.session_state.extracted_biomarkers = None
                st.success("All documents cleared.")
                st.rerun()
        else:
            st.info("No documents uploaded yet.")
        
        st.markdown("---")
        st.markdown("### Conversation History")
        
        total_messages = sum(len(msgs) for msgs in st.session_state.messages.values())
        st.markdown(f"**{total_messages}** messages across **{len(SPECIALISTS)}** specialists")
        
        # Show per-specialist counts
        with st.expander("Messages per specialist"):
            for spec_name, msgs in st.session_state.messages.items():
                st.write(f"• {spec_name}: {len(msgs)} messages")
        
        if st.button("🗑️ Clear All Conversations", type="secondary"):
            st.session_state.messages = {k: [] for k in SPECIALISTS.keys()}
            st.success("All conversations cleared.")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Reset Application")
        
        st.warning("This will clear all data including documents, conversations, and settings.")
        
        if st.button("🔄 Reset Everything", type="secondary"):
            # Keep only API key
            api_key = st.session_state.get('user_api_key')
            st.session_state.clear()
            if api_key:
                st.session_state.user_api_key = api_key
            st.rerun()
    
    # About Tab
    with settings_tabs[3]:
        st.markdown(f"""
        ### About {BRAND_NAME}
        
        **Version:** {VERSION}
        
        **Created by:** {CREATED_BY}
        
        ---
        
        #### Technology Stack
        
        - **AI Engine:** OpenAI GPT-4o
        - **Document Processing:** LangChain
        - **Vector Search:** FAISS (when available)
        - **Frontend:** Streamlit
        - **Visualization:** Plotly
        
        #### Features
        
        - 7 AI Specialist Consultants
        - 20+ Supported File Formats
        - LangChain Document Intelligence
        - Comprehensive Biomarker Tracking
        - Evidence-Based Wellness Protocols
        - Advanced Health Analytics
        
        #### Supported File Formats
        
        **Documents:** PDF, DOCX, DOC, TXT, RTF, ODT
        **Spreadsheets:** CSV, XLSX, XLS, TSV
        **Data:** JSON, XML, YAML
        **Web:** HTML, Markdown
        **Presentations:** PPTX, PPT
        **Images (OCR):** PNG, JPG, JPEG, WebP, GIF, BMP, TIFF
        **Health Data:** HL7, CCDA, FHIR
        
        ---
        
        #### Deployment
        
        This application is optimized for **Streamlit Community Cloud**.
        
        To deploy:
        1. Push code to GitHub
        2. Connect to Streamlit Cloud
        3. Add `OPENAI_API_KEY` to Secrets
        4. Deploy!
        
        ---
        
        #### Legal Disclaimer
        
        This platform is for **informational and educational purposes only**.
        
        {BRAND_NAME} does not provide medical diagnosis, treatment, or prescriptions.
        The AI specialists provide general health information based on current research
        and should not replace consultation with qualified healthcare professionals.
        
        Always consult with your physician before starting any new health protocol,
        supplement regimen, or treatment plan.
        
        ---
        
        © 2026 {BRAND_NAME} | {CREATED_BY} | All Rights Reserved
        """)


# =====================================================
# FOOTER
# =====================================================
st.markdown(
    f"""
<div class="agency-footer">
    <p style="font-size: 1.2rem; margin-bottom: 0.5rem; font-weight: 700;">
        🏥 {BRAND_NAME}
    </p>
    <p style="color: #64748B; margin-bottom: 0.75rem;">
        Powered by 🦜 LangChain & GPT-4 | Version {VERSION}
    </p>
    <p style="color: #94A3B8; margin-bottom: 1.5rem;">
        Developed by <strong>{CREATED_BY}</strong> | Premium Health Intelligence Platform
    </p>
    <p style="font-size: 0.7rem; color: #94A3B8; max-width: 900px; margin: 0 auto; line-height: 1.6;">
        <strong>DISCLAIMER:</strong> This platform is for informational and educational purposes only. 
        {BRAND_NAME} does not provide medical prescriptions, diagnosis, or treatment. 
        The AI specialists provide general health information based on current research and should not replace 
        consultation with qualified healthcare professionals. Always consult with your physician before 
        starting any new health protocol, supplement, or treatment plan.
    </p>
    <p style="font-size: 0.65rem; color: #CBD5E1; margin-top: 1.5rem;">
        © 2026 {BRAND_NAME} | {CREATED_BY} | All Rights Reserved
    </p>
</div>
""",
    unsafe_allow_html=True,
)
