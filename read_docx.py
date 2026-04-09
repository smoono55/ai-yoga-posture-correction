import zipfile
import xml.etree.ElementTree as ET

def extract_text(docx_file):
    try:
        with zipfile.ZipFile(docx_file) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            
            # The namespace for w:t (text nodes)
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            texts = []
            for node in tree.findall('.//w:t', namespaces):
                if node.text:
                    texts.append(node.text)
            
            with open('yoga_study_guide_extracted.md', 'w', encoding='utf-8') as f:
                f.write('\n'.join(texts))
            print("Successfully extracted to yoga_study_guide_extracted.md")
    except Exception as e:
        print("Error reading docx:", e)

extract_text('yoga_study_guide.docx')
