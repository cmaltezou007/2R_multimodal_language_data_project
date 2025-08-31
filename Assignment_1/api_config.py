#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 17:55:24 2025

@author: constantina
"""

from openai import OpenAI
from google import genai


OPENAI_API_KEY = "" # I removed my own key from here. You should copy-paste your own key for this project to work during LLM tasks.

def get_openai_client():
 return OpenAI(api_key=OPENAI_API_KEY)
