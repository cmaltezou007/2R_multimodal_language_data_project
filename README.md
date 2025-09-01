# 2R_multimodal_language_data_project

This repository contains two assignment submissions for my 2R course in Python:
1. Short exercises for workspace management, OCR, speech recognition/transcription, image/video processing, and LLMs.
2. A bigger project with Sphinx-documentation (found in `documentation > build > latex > 2rassignment2.pdf`) aimed at identifying overall trust attitudes towards AI as expressed in public YouTube videos, sentiment polarity, and thematic patterns that characterise these attitudes, using a combination of computational and language-based approaches.

Both assignments require API keys to work, which you can copy-paste yours in their respective config files once you clone this repository.


## Assignment 2 project folder structure

-- 1. Assignment_2

---- 1.1. data

-------- 1.1.1. processed

---- 1.2. figures

---- 1.3. documentation (sphinx)

---- 1.4. custom_packages

-------- 1.4.1. preprocessing_modules

-------- 1.4.2. modelling_modules

-------- 1.4.3. misc_modules

---- 1.5. src


## Getting started

- Clone this repository.
- Restore the exact package versions I used in Assignment 2 (for reproducibility purposes) by typing `python -m pip install -r requirements.txt`.
- Remember to edit the respective assignment's config python file, to include your own API keys for OpenAI, Google Gemini and YouTube.
- If for whatever reason you want to update your cloned repository's Sphinx documentation (Assignment_2 only), simply type in the terminal `make -C documentation clean` while inside the base project dir, followed by `make -C documentation latexpdf`. However, you may have to install latexmk first by typing `sudo apt install latexmk`. You can then find the latex-generated PDF Sphinx documentation file in `documentation > build > latex > 2rassignment2.pdf`.
