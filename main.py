import gradio as gr
import pickle

from transformers import pipeline

with open('./model/log_regression.pkl', 'rb') as f:
	model = pickle.load(f) 

pipe = pipeline("text-classification", model=model)

def predict(text):
    return pipe(text)[0]

iface = gr.Interface(
  fn=predict, 
  inputs='text',
  outputs='text',
  examples=[["Glen told CNN that the letter arrived at the property a couple of years ago, but he has only recently taken it to the local historical society, so they can research it further.The envelope has a 1 pence stamp bearing the head of King George V. The letter was sent in the middle of World War I – more than a decade before Queen Elizabeth II was born."]]
)

iface.launch()
