import fitz  # PyMuPDF library for PDF processing
import pytesseract
from PIL import Image
import io
import time
import os
import google.generativeai as genai

# --- Manual Tesseract Path (Uncomment and modify if tesseract.exe is NOT in your system PATH) ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# -------------------------------------------------------------------------------------------------

# গ্লোবাল ভেরিয়েবল হিসেবে Gemini মডেল ইনিশিয়ালাইজ করা হবে, যাতে বারবার লোড না হয়
gemini_model = None

def initialize_gemini_model(api_key, model_name='gemini-1.5-flash-latest'):
    """Initializes the Google Gemini model."""
    global gemini_model
    if gemini_model is None:
        try:
            print(f"Configuring Google Gemini API Key and loading model '{model_name}'...")
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model_name)
            print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error initializing Gemini model '{model_name}': {e}")
            print("Please ensure your API Key is correct, the model name is valid for your region, and you have internet access.")
            gemini_model = None

def summarize_text_bangla(text, max_output_tokens=1000):
    """
    Summarizes the given Bengali text using Google Gemini API.
    """
    global gemini_model
    if gemini_model is None:
        return "Gemini model not initialized. Cannot summarize."

    if not text.strip():
        return "No text provided for summarization."

    print(f"Summarizing text using Gemini Pro (max_output_tokens={max_output_tokens})...")
    try:
        prompt = f"নিম্নলিখিত বাংলা টেক্সটটিকে সংক্ষিপ্ত করুন:\n\n{text}"
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens
            )
        )

        if response.text:
            return response.text
        else:
            feedback = response.prompt_feedback
            if feedback and feedback.block_reason:
                return (f"Summarization blocked due to safety concerns. "
                        f"Reason: {feedback.block_reason.name}. "
                        f"Safety Ratings: {feedback.safety_ratings}")
            else:
                return "Summarization failed: Gemini returned an empty response or unexpected format."

    except Exception as e:
        if "ResourceExhausted" in str(e) or "400" in str(e):
            return (f"Error during summarization: The input text is too long for the Gemini model's "
                    f"current context window (e.g., more than ~30,000 tokens for Pro model). "
                    f"Please try a shorter document or implement text chunking. Original error: {e}")
        return f"An unexpected error occurred during summarization: {e}"

# --- NEW FUNCTION FOR QUESTION ANSWERING ---
def ask_question_about_text(context_text, question, max_output_tokens=500):
    """
    Asks a question about the given context text using Google Gemini API.
    """
    global gemini_model
    if gemini_model is None:
        return "Gemini model not initialized. Cannot answer questions."

    if not context_text.strip():
        return "No context text provided to answer the question from."
    if not question.strip():
        return "No question provided."

    print(f"\nAsking question using Gemini Pro (max_output_tokens={max_output_tokens})...")
    try:
        # The prompt for question answering
        # It's crucial to structure the prompt to guide the model to answer based ONLY on the provided text.
        prompt = (f"নিম্নলিখিত টেক্সটটির উপর ভিত্তি করে, শুধুমাত্র এই টেক্সট থেকে উত্তর দিন। "
                  f"যদি টেক্সটে উত্তর না থাকে, তবে বলুন 'দুঃখিত, এই টেক্সটে এই প্রশ্নের উত্তর নেই।':\n\n"
                  f"টেক্সট:\n{context_text}\n\n"
                  f"প্রশ্ন: {question}\n\n"
                  f"উত্তর:")

        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens
            )
        )

        if response.text:
            return response.text
        else:
            feedback = response.prompt_feedback
            if feedback and feedback.block_reason:
                return (f"Question answering blocked due to safety concerns. "
                        f"Reason: {feedback.block_reason.name}. "
                        f"Safety Ratings: {feedback.safety_ratings}")
            else:
                return "Question answering failed: Gemini returned an empty response or unexpected format."

    except Exception as e:
        # Quota or content length issues
        if "ResourceExhausted" in str(e) or "400" in str(e):
            return (f"Error during question answering: The context text or generated answer is too long for the Gemini model's "
                    f"current context window (e.g., more than ~30,000 tokens for Pro model). "
                    f"Please try a shorter document or ask a more concise question. Original error: {e}")
        return f"An unexpected error occurred during question answering: {e}"


def extract_bangla_text_from_pdf(pdf_path, output_filename="extracted_bangla_text.txt", dpi=300):
    """
    Extracts Bangla text from all pages of a PDF, using OCR for image-based text,
    and saves the output to a specified text file. Each page's text will be preceded
    by a '--- Page X ---' header.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_filename (str): The name of the file to save the extracted text.
        dpi (int): Dots Per Inch for converting PDF pages to images. Higher DPI
                   improves OCR accuracy but increases processing time. (e.g., 300, 400).

    Returns:
        tuple: A tuple containing:
            - str: A success or error message.
            - str: The path to the output file (or None if error).
            - float: The total runtime in seconds (or None if error).
            - str: The full extracted text (for summarization/QA).
    """
    full_text_list = []
    start_time = time.time()
    error_message = None
    output_file_path = None
    extracted_full_text = ""

    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The PDF file was not found at: {pdf_path}")

        doc = fitz.open(pdf_path)

        print(f"Starting text extraction from '{os.path.basename(pdf_path)}'...")
        print(f"Processing {len(doc)} page(s)...")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            print(f"  Processing page {page_num + 1}...")

            full_text_list.append(f"--- Page {page_num + 1} ---\n")

            text_content = page.get_text("text")
            if text_content.strip():
                full_text_list.append(text_content)
            else:
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                ocr_text = pytesseract.image_to_string(img, lang='ben')
                full_text_list.append(ocr_text)

            full_text_list.append("\n\n")

        doc.close()
        extracted_full_text = "".join(full_text_list)

        output_file_path = output_filename
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(extracted_full_text)

        end_time = time.time()
        runtime = end_time - start_time
        message = f"OCR process complete. Extracted text saved to '{output_file_path}'."
        message += f"\nOCR Runtime: {runtime:.2f} seconds."

        return message, output_file_path, runtime, extracted_full_text

    except FileNotFoundError as e:
        error_message = str(e)
    except fitz.FileDataError:
        error_message = "Invalid or corrupted PDF file. Please check the file."
    except pytesseract.TesseractNotFoundError:
        error_message = "Tesseract OCR engine not found. Please ensure Tesseract is installed and added to your system's PATH, or specify its path in the script."
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"

    end_time = time.time()
    runtime = end_time - start_time
    return f"Error: {error_message}\nRuntime before error: {runtime:.2f} seconds.", None, runtime, ""


if __name__ == "__main__":
    # --- Configuration ---
    my_pdf_file = "3page.pdf" # Make sure this is your correct PDF file name
    my_output_file = "extracted_bangla_text_output.txt"
    my_summary_file = "summarized_bangla_text_output.txt"
    my_qa_output_file = "Youtubes_output.txt" # New file for QA

    ocr_dpi = 300

    # !!! IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual API Key !!!
    GEMINI_API_KEY = "AIzaSyD2S4KMR19vo6GGrqSyIwE1UsfHs9ZISE0"

    # !!! IMPORTANT: Choose the model that works for you. Use one from the list you got earlier. !!!
    # Example: 'models/gemini-1.5-pro-latest' or 'models/gemini-1.5-flash-latest'
    # 'gemini-pro' also worked for you, so you can keep it if you prefer.
    GEMINI_MODEL_TO_USE = 'models/gemini-1.5-flash-latest' # Or 'gemini-pro', 'models/gemini-1.5-flash-latest' etc.


    # --- Manual Tesseract Path (Uncomment if needed) ---
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # -------------------------------------------------------------------------------------------------

    # Step 1: Extract Text from PDF
    status_message_ocr, file_path_ocr, time_taken_ocr, full_extracted_text = extract_bangla_text_from_pdf(
        my_pdf_file, my_output_file, ocr_dpi
    )

    print("\n" + "="*50)
    print(status_message_ocr)
    print("="*50)

    if file_path_ocr:
        print(f"\nExtracted text content is available in the file: {os.path.abspath(file_path_ocr)}")

        # Initialize Gemini model only once
        initialize_gemini_model(GEMINI_API_KEY, model_name=GEMINI_MODEL_TO_USE)

        if gemini_model: # Proceed only if model was initialized successfully

            # --- Summarization Part (Optional) ---
            print("\nStarting summarization process using Google Gemini API...")
            summarized_text = summarize_text_bangla(full_extracted_text, max_output_tokens=1000)
            with open(my_summary_file, 'w', encoding='utf-8') as f:
                f.write(summarized_text)
            print(f"Summarization complete. Summary saved to '{my_summary_file}'.")
            print(f"Find your summarized text in the file located at: {os.path.abspath(my_summary_file)}")

            # --- Question Answering Part ---
            print("\n" + "="*50)
            print("Starting Question Answering Mode.")
            print("Type your questions (in Bangla) and press Enter. Type 'exit' to quit.")
            print("="*50)

            # নতুন সেশন শুরুর আগে ফাইলটি খালি করে নেওয়া
            with open(my_qa_output_file, 'w', encoding='utf-8') as f:
                f.write("--- Q&A Session ---\n\n")

            while True:
                user_question = input("\nআপনার প্রশ্ন (বাংলায়): ")
                if user_question.lower() == 'exit':
                    break

                if not full_extracted_text.strip():
                    print("Error: Extracted text is empty. Cannot answer questions.")
                    break

                answer = ask_question_about_text(full_extracted_text, user_question, max_output_tokens=500)
                
                # টার্মিনালে উত্তর দেখান
                print(f"\nউত্তর: {answer}")
                
                # প্রতিটি প্রশ্ন-উত্তর সাথে সাথে ফাইলে সেভ করুন
                with open(my_qa_output_file, 'a', encoding='utf-8') as f:
                    f.write(f"প্রশ্ন: {user_question}\nউত্তর: {answer}\n\n---\n\n")

            print(f"\nAll questions and answers have been saved to '{my_qa_output_file}'.")
            print(f"Find your Q&A output in the file located at: {os.path.abspath(my_qa_output_file)}")

        else:
            print("\nGemini model could not be initialized. Summarization and QA skipped.")
    else:
        print("\nCould not proceed to summarization and QA due to an error in text extraction.")