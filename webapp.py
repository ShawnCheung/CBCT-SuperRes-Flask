"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
import time
import zipfile
from flask import Flask, render_template, request, send_file

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        files = request.files.getlist('fileFolder')
        subfolder = files[0].filename.split('/')[0]+str(time.time())

        os.makedirs("input/"+subfolder, exist_ok=True)
        os.makedirs("output/"+subfolder, exist_ok=True)
        for file in files:
            file.save("input/"+subfolder+"/"+file.filename.split('/')[1])
        os.system("python temp-cycle-gan/mytest_mat.py --results_dir {} --dataroot {}".format("output/"+subfolder, "input/"+subfolder))

        file_list = []
        for root, dirs, files in os.walk("output/"+subfolder):
            for file in files:
                file_list.append(os.path.join(root, file))
        dl_name = '{}.zip'.format(subfolder)
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
            for _file in file_list:
                with open(_file, 'rb') as fp:
                    zf.writestr(_file.split("\\")[1], fp.read())
        memory_file.seek(0)
        return send_file(memory_file, attachment_filename=dl_name, as_attachment=True)
    return render_template("test.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing cycle-GAN models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
