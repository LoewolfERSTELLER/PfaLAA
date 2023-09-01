from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import os
import numpy as np

def predict_answer(question, model, tokenizer, max_length):
    """
    Nutzen Sie das Modell, um eine Antwort basierend auf der gestellten Frage zu generieren.
    """
    # Frage tokenisieren
    seq = tokenizer.texts_to_sequences([question])
    
    # Sequenz auf die gleiche Länge bringen
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')

    # Antwort mit dem Modell vorhersagen
    predicted_seq = model.predict(padded_seq)
    
    # Vorhersage in Text umwandeln
    predicted_answer = []
    for word_prob in predicted_seq[0]:
        predicted_word_index = np.argmax(word_prob)
        if predicted_word_index == 0:  # Das ist unser Padding-Token
            break
        predicted_word = tokenizer.index_word.get(predicted_word_index, "Ich weiß es nicht.")
        predicted_answer.append(predicted_word)

    return ' '.join(predicted_answer) if predicted_answer else "Ich weiß es nicht."


# Modell, Tokenizer und max_length laden
model_dir = "PfaLAA"
model = load_model(os.path.join(model_dir, "model.keras"))

with open(os.path.join(model_dir, 'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(os.path.join(model_dir, 'max_length.txt'), 'r') as f:
    max_length = int(f.read().strip())

# Fragen an die KI stellen
while True:
    user_input = input("Frage: ")
    if user_input.lower() in ["ende", "exit", "beenden", "stop"]:
        print("Beenden...")
        break
    answer = predict_answer(user_input, model, tokenizer, max_length)
    print("Antwort:", answer)
