import os
import json 
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import DeepLake

load_dotenv()

# Initializes your app with your bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

#Setup RetrievalQA Chain 
embeddings = OpenAIEmbeddings()
db = DeepLake(dataset_path='<YOUR_DATASET>', embedding=embeddings, read_only=True)
retriever = db.as_retriever()
model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
qa = RetrievalQA.from_llm(llm=model, retriever=retriever, verbose=True)

@app.message("")
def message_handler(message, say):
        message_text = message['text']
        response = qa.run(message_text)
        say(response)
    
# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN")).start()