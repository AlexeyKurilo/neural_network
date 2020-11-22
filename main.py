import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import shutil
import neuron

app = FastAPI()


# @app.get('/')
# def home():
#     return {'key': 'Hello'}

#
# @app.post("/files/")
# async def create_file(file: bytes = File(...)):
#     return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    with open("neuron_file.png", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = int(neuron.start('neuron_file.png'))
    os.remove('neuron_file.png')
    # file.save(os.path.join(settings.UPLOAD_FOLDER, file.filename))
    # return file.filename
    return {'Result': result}


# @app.post("/CheckNeuron/")
# async def check_neuron(file: UploadFile = File(...)):
#     return File.filename


@app.get("/")
async def main():
    content = """
<body>
<h1>Simple neural network</h1>
<h1></h1>
<h5>The neural network is trained as much as possible on images with a resolution of 28x28, the percentage of 
recognition on these images is 99%.</h5>
<h1></h1>
<h5>If you load an image of a different extension, then unload the maximally square image so that there is no distortion.</h5>
<h1></h1>
<h5>Thank you for understanding.</h5>
<form action="/uploadfile/" enctype="multipart/form-data" method="post">
<input name="file" type="file">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
