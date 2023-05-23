from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
zipurl = 'https://drive.google.com/uc?export=download&confirm=no_antivirus&id=12qd4U4gqTg3igWkL6LjlJYdW49In7IrX'

with urlopen(zipurl) as zipresp:
    with ZipFile(BytesIO(zipresp.read())) as zfile:
        zfile.extractall('./')