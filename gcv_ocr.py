# pip install google-cloud-vision
import cv2
import time



def detect_text(path):
    """Detects text in the file."""
    import os
    import io
    from google.cloud import vision
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ghostwriterocr-f95f43035269.json"
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts


def write_ON_File(texts,timestr,word=False):
    path = "document/ocr/"
    # print(texts)
    if word:
        # pip install python-docx
        if len(texts) > 0:
            from docx import Document
            document = Document()
            document.add_paragraph(texts[0].description)
            document.save(path+"word/ocr_output_{}.docx".format(timestr))
    else:
        if len(texts) > 0:
            f = open(path + "text/ocr_output_{}.txt".format(timestr), "w+")
            f.write(texts[0].description)
            f.close()


def main():
    pass
   # text = detect_text("document/image/image_20190617_1845.png")
   # print(text)
   # write_ON_File(text,True)
   # write_ON_File(text)


# if __name__ == "__main__":
#     main()