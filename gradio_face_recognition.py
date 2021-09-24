import gradio as gr
from face_recog import detect_face

title = "Face Recognition"
desc = "--"
# examples = [
#     ["example_images/3.png"],
#     ["example_images/2.png"],
#     ["example_images/1.png"],
# ]

examples = [
    ["images/taylor.jpg"],
    ["images/selena.jpg"],
    ["Taylor Swift"],
    ["Selena Gomez"],
    ["images/taylorselena.jpg"]
]
inputs = [gr.inputs.Image(label="Picture of Person 1", type="file"),
          gr.inputs.Image(label="Picture of Person 2", type="file"),
          gr.inputs.Textbox(label="Name of Person 1"),
          gr.inputs.Textbox(label="Name of Person 2"),
          gr.inputs.Image(label="Picture of Person 1 and 2", type="file")
        ]
outputs = [gr.outputs.Image(label="Predicted Face", type="pil")]
gr.Interface(fn=detect_face, inputs=inputs, outputs=outputs, title=title, description=desc, 
             #examples=examples,
             allow_flagging=False, server_name="0.0.0.0", server_port=8000).launch()