import re
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import pickle

def convert(filename, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams = LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(filename, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text

if __name__ == '__main__':
    dictionary = {}
    p = 0
    while True:
        print(p)
        text = convert('/home/liuyu/Desktop/bv_cvxbook.pdf', range(p,p+10))
        if text == '':
            break
        text = re.sub('[^0-9a-zA-Z]+', ' ', text)
        item = text.split(' ')
        for word in item:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
        p += 10
    pickle.dump(dictionary, open('count.pkl','wb'))
