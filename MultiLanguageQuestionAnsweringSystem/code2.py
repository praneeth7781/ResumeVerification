from transformers import pipeline
from googletrans import Translator
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

# pip install googletrans==3.1.0a0

# tokenizer = AutoTokenizer.from_pretrained("salesken/translation-hi-en")
# model = AutoModelForSeq2SeqLM.from_pretrained("salesken/translation-hi-en")


model_checkpoint = "hashtag7781/bert-finetuned-squad"
question_answerer = pipeline("question-answering", model=model_checkpoint)


def predict(Context, Context_Language, Question, Question_Language):
    if Context_Language == "Telugu":
        translator = Translator()
        translated_context = translator.translate(
            Context, src='te', dest='en').text

    elif Context_Language == "Hindi":
        translator = Translator()
        translated_context = translator.translate(
            Context, src='hi', dest='en').text

    else:
        translator = Translator()
        translated_context = Context

    if Question_Language == "Telugu":
        translator = Translator()
        text_to_translate = translator.translate(Question,
                                                 src='te',
                                                 dest='en')
        translated = text_to_translate.text
        ans = question_answerer(question=translated,
                                context=translated_context)
        print(translated)
        return ans['answer']
    elif Question_Language == "Hindi":
        # hin_snippet = Question
        # inputs = tokenizer.encode(
        #     hin_snippet, return_tensors="pt", padding=True, max_length=512, truncation=True)
        # outputs = model.generate(
        #     inputs, max_length=128, num_beams=None, early_stopping=True)
        # translated = tokenizer.decode(outputs[0]).replace(
        #     '<pad>', "").strip().lower()
        translator = Translator()
        text_to_translate = translator.translate(Question,
                                                 src='hi',
                                                 dest='en')
        translated = text_to_translate.text
        ans = question_answerer(question=translated,
                                context=translated_context)
        print(translated)
        return ans['answer']
    else:
        ans = question_answerer(question=Question, context=translated_context)
        print("english")
        return ans['answer']


iface = gr.Interface(
    fn=predict,
    inputs=['text', gr.Dropdown(["Telugu", "Hindi", "English"]), 'text', gr.Dropdown(
        ["Telugu", "Hindi", "English"])],
    outputs='text',
    allow_flagging='never',
)

iface.launch(share=True)


# p = pipeline('salesken/translation-hi-en')


# def transcribe(audio):
#     text = p(audio)["text"]
#     return text


# gr.Interface(
#     fn=transcribe,
#     inputs=gr.Audio(source="microphone", type="filepath"),
#     outputs="text").launch()


# hin_snippet = "कोविड के कारण हमने अपने ऋण ब्याज को कम कर दिया है"


# inputs = tokenizer.encode(
#     hin_snippet, return_tensors="pt",padding=True,max_length=512,truncation=True)

# outputs = model.generate(
#     inputs, max_length=128, num_beams=None, early_stopping=True)

# translated = tokenizer.decode(outputs[0]).replace('<pad>',"").strip().lower()
# print(translated)
