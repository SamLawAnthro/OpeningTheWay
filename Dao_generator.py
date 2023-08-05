import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re
import torch

#loads text file, read it and assigns the string to a variable
f = open("tao.txt","r")
tao_text=f.read()

#loads text file, read it and assigns the string to a variable
f = open("tao.txt","r")
tao_text=f.read()

# pre-process the text into a list where each item is a chapter (the base unit which we want to analyse similarity)
def split_book_string(book_string):
    split_lines = re.split(r'\d+\s*-\s*', book_string)
    non_empty_lines = [line for line in split_lines if line.strip()]
    return non_empty_lines
    
tao_split = split_book_string(tao_text)

#for each document, create a tensor embedding by using the sentence-embedding library
sbert_model = SentenceTransformer('stsb-roberta-large')
document_embeddings = sbert_model.encode(tao_split, convert_to_tensor=True)


def generate_chapter(document_embeddings, sbert_model, book_split, prompt_embedding): 

    # calculuates cosine similarity of user response v. document embeddings from earlier
    # calculate the cosine similarity between the prompt_embedding and each of the document embeddings, returns a tensor that is the size
    # of documents in document_embeddings, with each index being the similiarity between prompt_embedding and document_embedding
    
    cosine_scores = util.pytorch_cos_sim(prompt_embedding, document_embeddings)
    
    # finds most similar document and returns it. 
    # The largest number in the tensor will be the most similar passage, so the index of that value will be the chapter we want
    chosen_chapter = torch.argmax(cosine_scores).item()
    return book_split[chosen_chapter]


def main():
    # Title and header
    st.title("Opening the Way")
    st.header("A clear stream of water runs, from poem to poem, wearing down the indestructible, finding the way around everything that obstructs the way. Good drinking water. The way is funny, keen, kind, modest, indestructibly outrageous, and inexhaustibly refreshing. Of all the deep springs, this is the purest water. To me, it is also the deepest spring. -- Ursula K. Le Guin")
    st.write("Tell us about your thirst, about the sharp edges to be worn down, what obstructs you or eddies you are stuck in. Ask a question, describe a problem, tell a story. Questioning builds a Way.!")

    # Get an input from the user
    prompt = st.text_input("Writing opens the Way:")

    #  Embeds the user's response 
    prompt_embedding = sbert_model.encode(prompt, convert_to_tensor=True)


    if prompt:
        # Generate and display the most relevant chapter
        chapter = generate_chapter(document_embeddings, sbert_model, tao_split, prompt_embedding)
        st.subheader("Lao Tzu says:")
        st.write(chapter)
if __name__ == "__main__":
    main()
