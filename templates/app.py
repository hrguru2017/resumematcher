import os
import fitz
import docx
import io
import json 
from flask import Flask, render_template, request, flash, redirect, url_for, current_app
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai

# --- Path Definitions & Debugging ---
# Get the absolute path of the directory where this file (app.py) is located.
project_root = os.path.dirname(os.path.abspath(__file__))
# Define the absolute path to the templates folder.
template_folder = os.path.join(project_root, 'templates')
# Define the absolute path to the static folder.
static_folder = os.path.join(project_root, 'static')

# --- Add a print statement to see what path is being calculated in the Render environment ---
print(f"--- SERVER STARTUP DEBUG ---")
print(f"Project Root Path: {project_root}")
print(f"Template Folder Path: {template_folder}")
print(f"Static Folder Path: {static_folder}")
print(f"--- END SERVER STARTUP DEBUG ---")


load_dotenv() 

try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except AttributeError:
    print("ERROR: Google AI API key not configured. Please set GOOGLE_API_KEY in your .env file.")

# Create the Flask app instance, explicitly telling it where to find the templates and static files.
app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_for_development')

# --- Helper functions for text extraction ---
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ... (The rest of your app.py file remains exactly the same) ...
# ... (extract_text_from_pdf, extract_text_from_docx, analyze_jd_with_llm, etc.) ...
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        current_app.logger.error(f"Error reading PDF: {e}")
        return None
    return text
def extract_text_from_docx(file_bytes):
    text = ""
    try:
        in_memory_stream = io.BytesIO(file_bytes)
        doc = docx.Document(in_memory_stream)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        current_app.logger.error(f"Error reading DOCX: {e}")
        return None
    return text

def analyze_jd_with_llm(jd_text):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""
    You are an expert HR analyst... The JSON object must have the following keys:
    - "job_title": string
    - "company_name": string
    - "job_locations": list of strings
    - "employment_type": string
    - "experience_required": string
    - "leadership_experience": string or null
    - "technical_skills": list of strings
    - "domain_knowledge": list of strings
    - "educational_requirements": string
    - "top_responsibilities": list of strings
    - "role_persona": string

    Instructions for specific keys:
    - For "top_responsibilities": Summarize the primary duties into a list of the top 5-6 most important points.
    - For "role_persona": Write a 2-3 sentence summary... This summary **must** weave in the 2-3 most critical technical skills...
    
    Job Description Text: ---
    {jd_text}
    ---
    
    IMPORTANT: Your entire response must be ONLY the raw text of a valid JSON object, starting with `{{` and ending with `}}`.
    """
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        results = json.loads(json_text)
        return results
    except Exception as e:
        current_app.logger.error(f"Error during AI JD analysis or JSON parsing: {e}")
        raw_response_text = "No response"
        if 'response' in locals() and hasattr(response, 'text'):
            raw_response_text = response.text
        current_app.logger.error(f"LLM Response Text (JD) was: {raw_response_text}")
        return {"error": str(e), "raw_response": raw_response_text}

def generate_boolean_search(analysis_results):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    relevant_info = {
        "job_title": analysis_results.get("job_title"),
        "technical_skills": analysis_results.get("technical_skills", []),
        "top_responsibilities": analysis_results.get("top_responsibilities", [])
    }
    info_json_str = json.dumps(relevant_info, indent=2)
    prompt = f"""
    You are an expert technical sourcer... Based on the following structured job details, generate a single, concise, and effective Boolean search string...
    (rest of prompt as before)
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        current_app.logger.error(f"Error generating boolean search: {e}")
        return "Error creating boolean search query."

def generate_recruiter_script(analysis_results):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    results_json_str = json.dumps(analysis_results, indent=2)
    prompt = f"""
    You are an expert hiring consultant creating a script for a recruiter... Based on the following structured job details, generate a list of 4-5 simple screening questions...
    (rest of prompt as before)
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        current_app.logger.error(f"Error during script generation: {e}")
        return "There was an error generating the screening questions."

def analyze_resume_with_llm(resume_text):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""
    You are an expert HR recruitment specialist, skilled at parsing resumes. Analyze the following resume text and extract the specified information in a valid JSON format.

    The JSON object must have these two top-level keys: "work_experience" and "top_technical_skills".

    1.  For "work_experience", the value must be a list of objects in reverse chronological order (most recent first). Each object must have these keys: "company_name", "job_title", "duration", "responsibilities" (as a list of strings).
    2.  For "top_technical_skills", the value must be a list of the 5 most prominent technical skills mentioned within the work experience sections.

    Here is the resume text:
    ---
    {resume_text}
    ---

    IMPORTANT: Your entire response must be ONLY the raw text of a valid JSON object, starting with `{{` and ending with `}}`. Do not include any other text, explanations, or markdown formatting.
    """
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        results = json.loads(json_text)
        return results
    except Exception as e:
        current_app.logger.error(f"Error during AI RESUME analysis or JSON parsing: {e}")
        raw_response_text = "No response"
        if 'response' in locals() and hasattr(response, 'text'):
            raw_response_text = response.text
        current_app.logger.error(f"LLM Response Text (Resume) was: {raw_response_text}")
        return {"error": str(e), "raw_response": raw_response_text}


def compare_jd_to_resume(jd_results, resume_results):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    jd_json_str = json.dumps(jd_results, indent=2)
    resume_json_str = json.dumps(resume_results, indent=2)
    prompt = f"""
    You are an expert senior technical recruiter... (rest of comparison prompt as before)
    """
    try:
        response = model.generate_content(prompt)
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        results = json.loads(json_text)
        return results
    except Exception as e:
        current_app.logger.error(f"Error during comparison AI analysis: {e}")
        raw_response_text = "No response"
        if 'response' in locals() and hasattr(response, 'text'):
            raw_response_text = response.text
        current_app.logger.error(f"LLM Response Text (Comparison) was: {raw_response_text}")
        return {"error": str(e), "raw_response": raw_response_text}

# --- Main Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        extracted_text = None
        pasted_text = request.form.get('jd_text', '').strip()
        if pasted_text: extracted_text = pasted_text
        elif 'jd_file' in request.files:
            file = request.files['jd_file']
            if file.filename != '' and file and allowed_file(file.filename):
                file_bytes = file.read()
                if file.filename.endswith('.pdf'): extracted_text = extract_text_from_pdf(file_bytes)
                elif file.filename.endswith('.docx'): extracted_text = extract_text_from_docx(file_bytes)
                if extracted_text is None:
                    flash(f"Could not read the content of the file: {secure_filename(file.filename)}.", 'danger')
                    return redirect(request.url)
            elif file.filename != '':
                flash('Invalid file type. Please upload a PDF or DOCX file.', 'danger')
                return redirect(request.url)
        
        if extracted_text:
            flash("Text extracted. Now analyzing with AI...", 'info')
            analysis_results = analyze_jd_with_llm(extracted_text)
            if analysis_results and "error" not in analysis_results:
                analysis_results['boolean_search_query'] = generate_boolean_search(analysis_results)
                analysis_results['recruiter_script'] = generate_recruiter_script(analysis_results)
                return render_template('results.html', results=analysis_results)
            else:
                flash(f"An error occurred during AI analysis. Please check server logs.", 'danger')
                if analysis_results and 'raw_response' in analysis_results:
                    current_app.logger.error(f"Problematic AI response: {analysis_results['raw_response']}")
                return redirect(url_for('index'))
        else: 
            if not pasted_text: flash('Please paste text or upload a valid PDF/DOCX file to analyze.', 'danger')
        return redirect(url_for('index'))
    return render_template('index.html')


@app.route('/resume', methods=['GET', 'POST'])
def resume_analyzer():
    if request.method == 'POST':
        extracted_text = None
        pasted_text = request.form.get('resume_text', '').strip()
        if pasted_text: extracted_text = pasted_text
        elif 'resume_file' in request.files:
            file = request.files['resume_file']
            if file.filename != '' and file and allowed_file(file.filename):
                file_bytes = file.read()
                if file.filename.endswith('.pdf'): extracted_text = extract_text_from_pdf(file_bytes)
                elif file.filename.endswith('.docx'): extracted_text = extract_text_from_docx(file_bytes)
                if extracted_text is None:
                    flash(f"Could not read the content of the file: {secure_filename(file.filename)}.", 'danger')
                    return redirect(request.url)
            elif file.filename != '':
                flash('Invalid file type. Please upload a PDF or DOCX file.', 'danger')
                return redirect(request.url)
        
        if extracted_text:
            flash("Resume text extracted. Now analyzing with AI...", "info")
            analysis_results = analyze_resume_with_llm(extracted_text)
            if analysis_results and "error" not in analysis_results:
                flash("Resume analysis complete!", "success")
                return render_template('resume_results.html', results=analysis_results)
            else:
                flash(f"An error occurred during AI analysis. Please check server logs.", 'danger')
                if analysis_results and 'raw_response' in analysis_results:
                    current_app.logger.error(f"Problematic AI response: {analysis_results['raw_response']}")
                return redirect(url_for('resume_analyzer'))
        else:
            flash('Please paste resume text or upload a valid PDF/DOCX file.', 'danger')
        return redirect(url_for('resume_analyzer'))
    return render_template('resume_analyzer.html')


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        jd_text = None
        if request.form.get('jd_text', '').strip():
            jd_text = request.form.get('jd_text').strip()
        elif 'jd_file' in request.files and request.files['jd_file'].filename != '':
            jd_file = request.files['jd_file']
            if allowed_file(jd_file.filename):
                jd_bytes = jd_file.read()
                if jd_file.filename.endswith('.pdf'): jd_text = extract_text_from_pdf(jd_bytes)
                elif jd_file.filename.endswith('.docx'): jd_text = extract_text_from_docx(jd_bytes)
        
        resume_text = None
        if request.form.get('resume_text', '').strip():
            resume_text = request.form.get('resume_text').strip()
        elif 'resume_file' in request.files and request.files['resume_file'].filename != '':
            resume_file = request.files['resume_file']
            if allowed_file(resume_file.filename):
                resume_bytes = resume_file.read()
                if resume_file.filename.endswith('.pdf'): resume_text = extract_text_from_pdf(resume_bytes)
                elif resume_file.filename.endswith('.docx'): resume_text = extract_text_from_docx(resume_bytes)

        if not jd_text or not resume_text:
            flash("Please provide both a Job Description and a Resume to compare.", "danger")
            return redirect(url_for('compare'))
        
        flash("Both documents received. Analyzing JD and Resume...", "info")
        
        jd_results = analyze_jd_with_llm(jd_text)
        if "error" in jd_results:
            flash(f"Error analyzing Job Description. Please try again.", "danger")
            current_app.logger.error(f"JD Analysis Error: {jd_results['error']}")
            return redirect(url_for('compare'))
            
        resume_results = analyze_resume_with_llm(resume_text)
        if "error" in resume_results:
            flash(f"Error analyzing Resume. Please try again.", "danger")
            current_app.logger.error(f"Resume Analysis Error: {resume_results['error']}")
            return redirect(url_for('compare'))

        flash("Analyses complete. Now performing comparison...", "info")
        comparison_results = compare_jd_to_resume(jd_results, resume_results)
        if "error" in comparison_results:
            flash(f"Error performing comparison. Please try again.", "danger")
            current_app.logger.error(f"Comparison Error: {comparison_results['error']}")
            return redirect(url_for('compare'))
        
        flash("Comparison complete!", "success")
        
        candidate_name = "Candidate"
        if resume_results.get('work_experience') and len(resume_results['work_experience']) > 0:
             # Heuristic: try to find a 'name' field if the LLM adds it, or use a placeholder
             # Since we don't explicitly ask for candidate name, we'll stick to a generic title.
             pass

        return render_template('compare_results.html', 
                               comparison=comparison_results, 
                               jd_title=jd_results.get('job_title', 'Job Description'),
                               candidate_name=candidate_name)

    return render_template('compare.html')

if __name__ == '__main__':
    app.run(debug=True)