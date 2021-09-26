import gradio as gr
from face_recog import detect_face

title = "Face Recognition"
desc = "Upload the pictures of 2 people you already know and also upload the picture that you want to identify"

examples = [["images/taylor.jpg", "Taylor Swift", "images/selena.jpg", "Selena Gomez", "images/taylorselena.jpg"]]

inputs = [gr.inputs.Image(label="Picture of Person 1", type="file"),
          gr.inputs.Textbox(label="Name of Person 1"),
          gr.inputs.Image(label="Picture of Person 2", type="file"),
          gr.inputs.Textbox(label="Name of Person 2"),
          gr.inputs.Image(label="Picture of Person 1 and 2", type="file")
         ]
outputs = gr.outputs.Image(label="Result", type="pil")
gr.Interface(fn=detect_face, inputs=inputs, outputs=outputs, title=title, description=desc,
             examples=examples).launch()
