import re
from io import BytesIO
from xml.sax.saxutils import escape

from docx import Document
from flask import Flask, redirect, render_template, request, send_file, url_for
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Maximum upload size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf", "docx"}

SKILLS = [
    "python", "machine learning", "deep learning",
    "artificial intelligence", "nlp", "natural language processing",
    "sql", "flask", "django", "java", "c++",
    "data analysis", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "html", "css", "javascript",
    "git", "github", "communication", "teamwork"
]

ATS_SECTIONS = [
    {
        "name": "Contact Information",
        "weight": 20,
        "keywords": []
    },
    {
        "name": "Education",
        "weight": 20,
        "keywords": [
            "education", "academic background", "qualification",
            "university", "degree"
        ]
    },
    {
        "name": "Work Experience",
        "weight": 20,
        "keywords": [
            "experience", "work history", "employment history",
            "professional experience", "internship"
        ]
    },
    {
        "name": "Skills",
        "weight": 15,
        "keywords": [
            "skills", "technical skills", "core competencies"
        ]
    },
    {
        "name": "Projects",
        "weight": 15,
        "keywords": [
            "projects", "academic projects", "personal projects"
        ]
    },
    {
        "name": "Certifications",
        "weight": 10,
        "keywords": [
            "certifications", "certificates", "courses",
            "training"
        ]
    }
]


def get_extension(filename):
    """Return the lowercase file extension without the dot."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def extract_pdf_text(file):
    reader = PdfReader(file)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages)


def extract_docx_text(file):
    """Read text from a Word document: body, tables, headers and footers."""
    document = Document(file)
    parts = []

    # Headers and footers often contain contact details
    for section in document.sections:
        for paragraph in section.header.paragraphs:
            parts.append(paragraph.text)
        for paragraph in section.footer.paragraphs:
            parts.append(paragraph.text)

    # Main body paragraphs
    for paragraph in document.paragraphs:
        parts.append(paragraph.text)

    # Tables (many resume templates use tables for layout)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)

    return "\n".join(part for part in parts if part.strip())


def extract_resume_text(file, extension):
    if extension == "pdf":
        return extract_pdf_text(file)

    if extension == "docx":
        return extract_docx_text(file)

    return ""


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_skills(text):
    text = clean_text(text)
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return sorted(found)


def match_label(score):
    """Short, honest label for the text-similarity score."""
    if score >= 40:
        return "High overlap"
    if score >= 20:
        return "Moderate overlap"
    return "Low overlap"


def check_ats_sections(resume_text):
    text = resume_text.lower()
    checks = []
    ats_score = 0
    ats_recommendations = []

    for section in ATS_SECTIONS:
        if section["name"] == "Contact Information":
            email_found = bool(
                re.search(
                    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                    resume_text,
                    re.IGNORECASE
                )
            )

            phone_found = bool(
                re.search(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", resume_text)
            )

            linkedin_found = "linkedin.com" in text

            present = email_found or phone_found or linkedin_found

        else:
            present = any(
                keyword in text
                for keyword in section["keywords"]
            )

        checks.append({
            "name": section["name"],
            "present": present,
            "weight": section["weight"]
        })

        if present:
            ats_score += section["weight"]
        else:
            ats_recommendations.append(
                f"Consider adding a clearly labelled "
                f"{section['name']} section if relevant to your background."
            )

    ats_score = min(100, ats_score)

    if ats_score < 100:
        ats_recommendations.append(
            "Use clear, standard section headings and keep the resume "
            "layout simple and easy to read."
        )

    return {
        "ats_score": ats_score,
        "ats_sections": checks,
        "ats_recommendations": ats_recommendations
    }


def analyze_resume(resume_text, job_description):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)

    if not resume_clean or not job_clean:
        return None

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(
        [resume_clean, job_clean]
    )

    score = cosine_similarity(
        vectors[0:1], vectors[1:2]
    )[0][0]

    match_score = round(float(score) * 100, 2)

    resume_skills = find_skills(resume_text)
    job_skills = find_skills(job_description)

    matched_skills = [
        skill for skill in job_skills
        if skill in resume_skills
    ]

    missing_skills = [
        skill for skill in job_skills
        if skill not in resume_skills
    ]

    recommendations = [
        f"Consider adding {skill} to your resume if you have this skill."
        for skill in missing_skills
    ]

    ats_result = check_ats_sections(resume_text)

    return {
        "match_score": match_score,
        "match_label": match_label(match_score),
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommendations": recommendations,
        "ats_score": ats_result["ats_score"],
        "ats_sections": ats_result["ats_sections"],
        "ats_recommendations": ats_result["ats_recommendations"]
    }


@app.route("/")
def home():
    return render_template("index.html", error=None, job_description="")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    # Opening /analyze directly (GET) just goes back to the upload page
    if request.method == "GET":
        return redirect(url_for("home"))

    error = None
    result = None

    uploaded_file = request.files.get("resume")
    job_description = request.form.get("job_description", "").strip()

    if not uploaded_file or not uploaded_file.filename:
        error = "Please upload your resume (PDF or Word .docx)."

    else:
        extension = get_extension(uploaded_file.filename)

        if extension not in ALLOWED_EXTENSIONS:
            if extension == "doc":
                error = (
                    "Old Word .doc files are not supported. "
                    "Please save the file as .docx or PDF and try again."
                )
            else:
                error = "Only PDF and Word (.docx) resumes are supported."

        elif not job_description:
            error = "Please enter a job description."

        else:
            resume_text = ""

            file_bytes = uploaded_file.read()

            if not file_bytes:
                error = (
                    "This file is empty (0 KB), so nothing was uploaded. "
                    "Please check that the file opens on your computer "
                    "(if it is in OneDrive or Google Drive, download it "
                    "first) and select it again."
                )

            try:
                if not error:
                    resume_text = extract_resume_text(
                        BytesIO(file_bytes), extension
                    )
            except Exception as exc:
                # Full details are printed in the terminal for debugging
                app.logger.exception("Could not read resume file")
                error = (
                    "Could not read this file. "
                    f"({type(exc).__name__}: {exc}) "
                    "If it is a PDF, make sure it is not password-protected "
                    "and try re-saving it from Word as a new PDF."
                )

            if not error:
                if not resume_text.strip():
                    error = (
                        "No readable text found in this file. "
                        "Please use a text-based PDF or a Word document."
                    )
                else:
                    try:
                        result = analyze_resume(resume_text, job_description)

                        if result is None:
                            error = "Could not analyze the provided text."
                    except Exception as exc:
                        app.logger.exception("Analysis failed")
                        error = (
                            "The resume was read, but the analysis failed. "
                            f"({type(exc).__name__}: {exc})"
                        )

    if error:
        # Stay on the upload page and show the error
        return render_template(
            "index.html", error=error, job_description=job_description
        )

    return render_template(
        "results.html",
        result=result,
        file_name=uploaded_file.filename
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        error="File is too large. Please upload a resume under 5 MB.",
        job_description=""
    ), 413


@app.route("/download-report", methods=["POST"])
def download_report():
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        title="Resume Analysis Report"
    )

    styles = getSampleStyleSheet()

    match_score = escape(request.form.get("match_score", "0"))
    ats_score = escape(request.form.get("ats_score", "0"))

    story = [
        Paragraph("AI Resume Analyzer Report", styles["Title"]),
        Spacer(1, 15),
        Paragraph(
            f"Resume Match Score: {match_score}%",
            styles["Heading2"]
        ),
        Paragraph(
            f"ATS Checklist Score: {ats_score}%",
            styles["Heading2"]
        ),
        Spacer(1, 12)
    ]

    sections = [
        ("Skills Found in Resume", "resume_skills"),
        ("Skills Required by Job", "job_skills"),
        ("Matched Skills", "matched_skills"),
        ("Missing Skills", "missing_skills"),
        ("Resume Improvement Suggestions", "recommendations"),
        ("ATS Section Checks", "ats_checks"),
        ("ATS Improvement Suggestions", "ats_recommendations")
    ]

    for heading, field_name in sections:
        story.append(
            Paragraph(escape(heading), styles["Heading2"])
        )

        items = request.form.getlist(field_name)

        if items:
            for item in items:
                story.append(
                    Paragraph(
                        "&#8226; " + escape(item),
                        styles["Normal"]
                    )
                )
                story.append(Spacer(1, 5))
        else:
            story.append(
                Paragraph("None", styles["Normal"])
            )

        story.append(Spacer(1, 10))

    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "Note: The ATS checklist score is based on this application's "
            "section checks. It is not an official score from an employer's "
            "Applicant Tracking System.",
            styles["Italic"]
        )
    )

    doc.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="resume_analysis_report.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(debug=True)
