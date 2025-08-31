import helper_functions
from pathlib import Path


base_data_folder_path = Path.cwd() / "assn1_data"


"""
For Task 1, I do the following steps:
1. Retrieve a list of full filepaths for all files inside the mixed media folder.
2.1. I pass that list to one of my helper functions, which will get all the relevant 
    info I need per file (file name, type, size, and dimensions - file type dependent).
2.2. The helper function will the copy each file to a relevant file type folder.
2.3.The helper function will then store that all that information in an output txt file.
3. Another helper function will check if each new file type folder exceeds 1MB in total size, and if it does it will
compress said folder into a zip file.

P.S. I know it wasn't part of this assignment, but I included error handling through try/except blocks around code areas
that could potentially throw an error, just to be able to follow better when/if something fails while debugging my code. 
"""
helper_functions.print_task_heading("Task 1. Workspace management")

input_folder = base_data_folder_path / "mixed_media"
output_filepath = base_data_folder_path / "files_metadata.txt"

list_of_filepaths = list(input_folder.iterdir())
helper_functions.copy_files_and_export_metadata(list_of_filepaths, output_filepath)

for organised_folder_name in ["text", "video", "audio", "images"]:
    helper_functions.create_zip_if_folder_exceeds_mb_limit(base_data_folder_path / organised_folder_name)


"""
For Task 2, I do the following steps:
1. Retrieve a list of full filepaths for all PDF files inside the tps folder.
2. I pass that list to one of my helper functions, which will convert each PDF document page into an image
    to be able to extract its text using OCR and then stores that form of transcription, along with a set of 3
    random sentences extracted from each transcription (for validation) into .txt files.
3. Repeats the process using the llama3.2-vision LLM instead of Tesseract OCR, for comparison.

Comments:
Tesseract OCR probably lacks the relevant language packs as it cannot transcribe Greek characters, its output in place 
of foreign Greek words is gibberish. It also seems to be struggling with stylised fonts such as italics or when all
characters of a word are in uppercase, and with landscape-orientation tables - giving incorrect words in their stead or complete 
gibberish.

In constrast, llama3.2-vision was more reliable.

"""
helper_functions.print_task_heading("Task 2.1. Tesseract Optical Character Recognition (OCR)")

input_folder = base_data_folder_path / "tps"
list_of_filepaths = list(input_folder.glob("*.pdf"))
helper_functions.export_text_from_image_pdfs(list_of_filepaths)

helper_functions.print_task_heading("Task 2.2. LLM Optical Character Recognition (OCR)")
helper_functions.export_text_from_image_pdfs_with_LLM(list_of_filepaths)


"""
For Task 3, I do the following steps:
1. Retrieve a list of full filepaths for all the audio files (mp3) inside the state-of-the-union folder.
2. I pass that list to one of my helper functions, which will transcribe each audio file and store its
    transcription in a .txt file. It will then compare the current transcription with the true transcript
    and display the word-error-rate and accuracy scores.
3. Repeats the process using the gpt-4o-transcribe LLM for comparison.
"""
helper_functions.print_task_heading("Task 3.1. Automatic Speech Recognition (ASR) and transcription")

input_folder = base_data_folder_path / "state-of-the-union"
list_of_filepaths = list(input_folder.glob("*.mp3"))
helper_functions.transcribe_audio(list_of_filepaths)

helper_functions.print_task_heading("Task 3.2. LLM Speech Recognition and transcription")
helper_functions.transcribe_audio_with_LLM(list_of_filepaths)


"""
Unfortunately, I did not know how to solve Task 4 for splicing out a 10-second video around a certain
point in the speech. Could you please provide solutions with your marking? Curious as to how it is done. 
Thank you!
"""



