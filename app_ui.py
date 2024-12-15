from flask import Flask, render_template, redirect, url_for
import webbrowser
import subprocess  # To run the main.py script

app = Flask(__name__)

def open_url(url):
    """Open the given URL in the default web browser."""
    webbrowser.open_new(url)

@app.route('/')
def index():
    functionalities = [
        ("OCR", "Harness the power of advanced OCR to extract text from scanned multilingual legal documents with exceptional accuracy. This tool seamlessly converts printed or handwritten text into editable English, enabling fast and efficient document digitization, even for large volumes of records."),
        ("NLP", "Let our NLP engine simplify the overwhelming complexity of legalese. This feature automatically detects and identifies essential details such as key clauses, names of involved parties, important dates, and contextual information. Save time by avoiding manual searches and ensure critical insights are always within reach."),
        ("Automated Clause Identification", "Lengthy legal documents can be daunting, but our platform makes them manageable by instantly locating and highlighting the most important clauses. Whether it's terms of agreement, indemnity clauses, or liability statements, you can pinpoint what matters most without wasting hours searching through pages."),
        ("Document Identification", "Eliminate confusion with our robust document tagging and categorization system. By intelligently segmenting legal documents, this feature ensures that different types of files are instantly recognizable, accurately organized, and readily accessible whenever needed. Stay efficient and stress-free with our streamlined approach"),
        ("Plain Language Translation", "Legal jargon can be challenging to navigate, but with our plain language translation feature, you can transform complex legal terminologies into simple, easy-to-understand text. Empower decision-making by breaking down sophisticated clauses into concise and actionable insights."),
        ("Plain Language Summaries", "For those needing a quick overview, our plain language summarization tool condenses detailed legal documents into short, clear summaries. This ensures you grasp the essence of the content without delving into the full document, saving valuable time and effort."),
    ]
    
    footer_links = {
        "My Resume": "https://drive.google.com/file/d/1yxXucZIxTqgCons7HOgQ7TgQtALdt8e7/view?usp=sharing",
        "Github": "https://github.com/nikhil8424",
        "Linkdin": "https://www.linkedin.com/in/nikhil-gupta-6b7705288/",
        
    }
    
    return render_template('app_ui_index.html', functionalities=functionalities, footer_links=footer_links)


@app.route('/get_started')
def get_started():
    """
    Executes the main.py script in a new subprocess to start the Tkinter application.
    This will run in a separate thread and open the application window independently.
    """
    try:
        subprocess.Popen(["python", "app.py"])
        # Redirect to show the user that the action was processed, could be a success message etc
        return redirect(url_for('index'))
    except FileNotFoundError:
        return "Error: app.py not found", 404


@app.route('/open_link/<path:url>')
def open_link(url):
    """Route to open URLs."""
    webbrowser.open_new(url)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True) # we will not specify the port as this is causing issues