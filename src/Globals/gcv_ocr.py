# pip install google-cloud-vision
import cv2
import time
import glob


def detect_text(path):
    """Detects text in the file."""
    import os
    import io
    from google.cloud import vision
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "auth/ghostwriterocr-f95f43035269.json"
    client = vision.ImageAnnotatorClient()
    texts = []
    print("Recognizing Words")
    for img in glob.glob(path + '/*.*'):
        with io.open(img, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)

        response = client.text_detection(image=image, image_context={"language_hints": ["en"]})
        text = response.text_annotations
        texts += [text]
    print("Recognizing Words completed")

    return texts


def write_on_file(texts, timestr, word=False):
    """Writes the output text in a txt file"""
    path = "output/ocr/"
    print("Saving documents #{}".format(len(texts)))
    if word:
        # pip install python-docx
        from docx import Document
        document = Document()

        for text in texts:
            if len(text) > 0:
                document.add_paragraph(u"{}".format(text[0].description))
        document.save(path+"word/ocr_output_{}.docx".format(timestr))
    else:
        f = open(path + "text/ocr_output_{}.txt".format(timestr), "w+")
        for text in texts:
            if len(text) > 0:
                f.writelines(u"{}".format(text[0].description))
        f.close()
    print("Saving output is completed")
