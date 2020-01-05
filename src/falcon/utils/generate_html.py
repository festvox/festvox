import sys
from xml.etree import ElementTree as ET

html = ET.Element('html')
body = ET.Element('body')
html.append(body)
div = ET.Element('div', attrib={'class': 'col-md-1'})
body.append(div)
audio = ET.Element('audio', attrib={'controls': 'controls'})
div.append(audio)
source = ET.Element('source', src='t.wav')
audio.append(source)

if sys.version_info < (3, 0, 0):
  # python 2
  ET.ElementTree(html).write(sys.stdout, encoding='utf-8',
                             method='html')
else:
  # python 3
  ET.ElementTree(html).write(sys.stdout, encoding='unicode',
                             method='html')
