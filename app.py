from flask import Flask, request, render_template, redirect, url_for,jsonify
import requests, urllib
from langchain_community.vectorstores.faiss import FAISS
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key='AIzaSyAclzx4NxvT7Cx9u7lyVE4lF57iiCVXxus', temperature=0.6)

# Initialize the model
# model = genai.GenerativeModel('gemini-pro')
# genai.configure(api_key='AIzaSyAclzx4NxvT7Cx9u7lyVE4lF57iiCVXxus')
vectordb = None
app = Flask(__name__)

def read_pdf_from_url(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        file = BytesIO(response.content)
        pdfreader = PdfReader(file)
        # read text from pdf
        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
    
    return raw_text

# Sample function to process a question
def process_question(question, pdf_url):

    retriever = vectordb.as_retriever()

    chain = RetrievalQA.from_chain_type(llm=llm,
                chain_type='stuff',
                retriever=retriever,
                input_key='query',
                return_source_documents=True)
    
    question += '. explain in very detail in at least 3 lines'

    print(chain(question))
    return chain(question)['result']

def search_arxiv(query, start=0, max_results=10):
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f"{query}"
    url = f'{base_url}search_query={search_query}&start={start}&max_results={max_results}'
    print(url)
    response = requests.get(url)
    
    if response.status_code == 200:
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            link = entry.find('{http://www.w3.org/2005/Atom}id').text
            paper_id = link.split('/abs/')[-1]
            pdf_link = f'http://arxiv.org/pdf/{paper_id}.pdf'
            papers.append({
                'title': title,
                'summary': summary,
                'link': pdf_link,
                'id': paper_id
            })
        return papers
    else:
        return []



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST','GET'])
def search():
    query = urllib.parse.quote(request.form.get('query', '').strip()) if request.method == 'POST' else request.args.get('query', '')
    page = int(request.args.get('page', 1))
    per_page = 10  # Number of results per page
    start = (page - 1) * per_page
    
    papers = search_arxiv(query, start=start, max_results=per_page)
    total_papers = 100  # Since the arXiv API max_results is 100, assuming max 100 results
    total_pages = (total_papers + per_page - 1) // per_page

    return render_template('results.html', papers=papers, page=page, total_pages=total_pages, query=query)

    

@app.route('/search/pdf/<index>', methods = ['GET','POST'])
def show_pdf(index):
    pdf_url = f'https://arxiv.org/pdf/{index}.pdf'
    
    return render_template('pdf.html', pdf_url=pdf_url,index=index)

@app.route('/search/pdf/form/<index>', methods = ['GET','POST'])
def form(index):
    question = request.json.get('question')
    answer = ''
    print(question)
    pdf_url = f'https://arxiv.org/pdf/{index}.pdf'
    answer = process_question(question, pdf_url)
    return jsonify({'answer':answer})

@app.route('/search/<index>', methods = ['GET','POST'])
def load_pdf_embeddings(index):
    pdf_url = f'https://arxiv.org/pdf/{index}.pdf'

    pdf_text = read_pdf_from_url(pdf_url)
    # We need to split the text using Character Text Split such that it sshould not increse token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 768,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(pdf_text)

    hf = HuggingFaceInstructEmbeddings()
    print('embeddings loaded')
    global vectordb

    vectordb = FAISS.from_texts(texts, hf)

    return redirect(url_for('show_pdf', index = index))



if __name__ == '__main__':
    app.run()
