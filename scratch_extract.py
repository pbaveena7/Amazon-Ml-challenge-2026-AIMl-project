import ast
import astor # might not be installed
import os
import re

file_path = r'c:\Users\NAVEEN\Desktop\Amazon-Entity-Resolution-AI\Amazon_Entity_Resolution_Improved_Complete.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

output = []
recording = False
for line in lines:
    if line.startswith('LEGAL_SUFFIXES =') or line.startswith('BLOCK_COLUMNS ='):
        recording = True
    
    if recording:
        output.append(line)
        
    if line.startswith('# ============================================================'):
        if '15. THRESHOLD SEARCH' in line or '6. LOAD GROUND TRUTH' in line or '11. BUILD VALIDATION CANDIDATE PAIRS' in line:
            recording = False

# We can just write a parser that captures the functions.
