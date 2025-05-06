import ollama

def get_model_response(question: str):
    try:
        response = ollama.chat(
            model="gemma3:4b",
            messages=[{"role": "system", "content": "Răspunde doar la întrebări despre plante."},
                      {"role": "user", "content": question}]
        )
        return response['message'].content

    except Exception as e:
        return f"Eroare: {str(e)}"
