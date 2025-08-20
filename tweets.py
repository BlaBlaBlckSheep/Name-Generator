import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI as Google
import os
from langchain import PromptTemplate, LLMChain

st.set_page_config(page_title='Name Generator', page_icon='🪺', layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Name Generator")

os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
model = Google(model = "gemini-1.5-flash-latest")

name_template = """
Give me {number} names on {topic} in {language}.
Please follow the below instructions:
1. Do not translate to English if the given language is not English.
2. If {language} is empty or not a real language, default to English and ignore {language}.
3. If {topic} is empty or does not make sense, then respond with "Please enter a topic for your name!".
4. Avoid unnecessary indents or spaces.
5. Keep the names funny and related to the theme or the {topic}
5.The names can be random and very creative but related to the topic.
Objective:
Generate humorous and imaginative names by combining inputs according to a set of rules. Each name should sound absurd, playful, and mildly ridiculous.

✅ Input Format

Each data point should include:

Input Elements:

A name of a pet or animal

A food item

An object found indoors

A human emotion or mood

A random sound or body noise (e.g., “snore”, “hiccup”)

Optional:

A noble or professional title (e.g., “Captain”, “Duchess”, “Dr.”)

🧾 Output Requirements

Produce a funny name using at least two of the input elements, following one of these patterns:

Name Patterns (choose one per output):

[Animal or Pet Name] + [Food Item]

Example: “Pickles Tacos”

[Emotion] + [Indoor Object]

Example: “Grumpy Blender”

[Title] + [Body Noise] + [Object]

Example: “Lord Snorty Desk”

[Pet Name] + [Body Noise] + [Food Item]

Example: “Mittens Burpy Pizza”

Encourage unexpected, non-logical combinations. The weirder and more pronounceable, the better.

✍️ Output Format Example
{
  "inputs": {
    "pet": "Fluffy",
    "food": "Spaghetti",
    "object": "Lamp",
    "emotion": "Sleepy",
    "sound": "Snort",
    "title": "Captain"
  },
  "funny_name": "Captain Snorty Lamp"
}
"""
name_prompt = NameTemplate(template = name_template, input_variables = ['number', 'topic', 'language'])

with st.form(key = 'name'):
    topic = st.text_input("Topic: ")
    number = st.number_input("Number of Names: ", value = 1, step = 1, max_value = 10, min_value = 1)
    language = st.text_input("Language: ")
    submit = st.form_submit_button("Generate")

if submit:
    name_chain = name_prompt | model
    response = name_chain.invoke({"number": number,
                                   "topic": topic,
                                   "language": language,})
    st.write(response.content)
