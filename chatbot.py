import gradio as gr
from langchain_ollama import OllamaLLM
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import difflib

# Charger le fichier texte nettoyé
txt_file_path = "C:\\Users\\Antarrio\\Desktop\\AGENT_IA\\RG_2024_2025_clean.txt"

def load_document():
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier texte: {e}")
        return ""

document = load_document()  # Charger le document initialement

# Charger le modèle Ollama (Machine Learning)
llm = None
try:
    llm = OllamaLLM(model="mistral:latest")
    print("Modèle Ollama chargé avec succès")
except Exception as e:
    print(f"Le modèle Ollama n'a pas pu être chargé. Vérifiez la configuration : {e}")

# Fonction pour rechercher une réponse dans le texte
def find_answer_from_text(query):
    global document
    document = load_document()  # Recharger le document à chaque requête
    lines = document.split("\n")
    query_lower = query.lower().strip()  # Normaliser la requête

    # Afficher les lignes du fichier pour vérification
    print("Lignes du fichier :")
    for line in lines:
        print(line)

    # Recherche simple pour vérifier la présence de la requête
    for line in lines:
        if query_lower in line.lower() or any(f"art. {i}" in line.lower() for i in range(70, 85)):
            print(f"Correspondance trouvée : {line}")
            return line

    # Si aucune correspondance exacte n'est trouvée, utiliser difflib
    closest_matches = difflib.get_close_matches(query_lower, [line.lower() for line in lines], n=5, cutoff=0.6)

    if closest_matches:
        context = "\n".join(closest_matches)
        print(f"Contexte trouvé dans le document :\n{context}")
        return context
    else:
        print("Aucune correspondance trouvée dans le document.")
        return None


# Fonction pour interagir avec le modèle et prioriser le texte
def chat_with_ai(input_text):
    print(f"Input reçu: {input_text}")

    if input_text.startswith("RG-GODF"):
        query = input_text[len("RG-GODF"):].strip()
        if not query:
            return "Veuillez fournir une requête après le mot clé RG-GODF."

        print(f"Recherche dans le document pour la requête : {query}")
        context = find_answer_from_text(query)

        if context:
            if llm:
                try:
                    response = llm.invoke(f"Basé sur ce contexte : {context}, répondez à la question : {query} (en français)")
                    print(f"Réponse générée : {response}")
                    return f"Réponse générée à partir du document : {response}"
                except Exception as e:
                    return f"Erreur avec le modèle Ollama : {e}"
            else:
                return "Le modèle Ollama n'est pas disponible pour répondre, mais j'ai recherché dans le fichier texte."
        else:
            return "Désolé, je n'ai pas trouvé de réponse précise dans le document. Essayez une autre formulation."

    # Si l'utilisateur ne commence pas par "RG-GODF", utiliser le modèle pour générer une réponse
    if not any(char in input_text for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        input_text = input_text + " (en français)"

    if llm:
        try:
            response = llm.invoke(input_text)
            print(f"Réponse du modèle Ollama : {response}")
            return f"Réponse du modèle Ollama : {response}"
        except Exception as e:
            return f"Erreur avec le modèle Ollama : {e}"
    else:
        return "Le modèle Ollama n'est pas disponible pour répondre."

# Interface Gradio pour l'interaction utilisateur
iface = gr.Interface(
    fn=chat_with_ai,
    inputs="text",
    outputs="text",
    title="Chat basé sur le document RG 2024-2025",
    description="Posez une question. Si elle concerne le Règlement Général, commencez par RG-GODF. Sinon, je répondrai de manière plus libre."
)

print("Lancement de l'interface Gradio...")
iface.launch(share=False, server_name="127.0.0.1", server_port=7862)
