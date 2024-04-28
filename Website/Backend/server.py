from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5TokenizerFast
import torch
import spacy
import google.generativeai as genai
from happytransformer import HappyTextToText, TTSettings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=5, min_length=1)

app = Flask(__name__)
CORS(app)
model_path_question = 'model.pth'
model_path_answer = 'answer_model.pth'

tokenizer = T5Tokenizer.from_pretrained('t5-base')
TOKENIZER = T5TokenizerFast.from_pretrained("t5-base")
device = "cpu"


question_model = torch.load(model_path_question, map_location=device)
answer_model = torch.load(model_path_answer, map_location=device)

@app.route('/home', methods=['POST'])
def handle_form_submission():
    try:
        data = request.get_json()
        user_paragraph = data['context']
        num_questions = int(data['Number'])
        max_answer_length = int(data['length'])
        print(user_paragraph)
        print(num_questions)
        print(max_answer_length)
        input_text = 'Generate Questions: '+ user_paragraph

        input_ids_q = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation='only_first',
            padding='max_length',
        ).input_ids.to(device)

        top_k = 90
        generated_questions = []
        Model_Given_Answer = []
        outputs = question_model.generate(
            input_ids_q,
            max_length=50,
            num_return_sequences=num_questions,
            do_sample=True,
            top_k=top_k,
            temperature=1.0
        )

        for i, sample_output in enumerate(outputs):
            generated_question = tokenizer.decode(sample_output, skip_special_tokens=True)
            generated_questions.append(generated_question)

        
        for i in range(num_questions):
            inputs = TOKENIZER(generated_questions[i], user_paragraph, max_length=256, padding="max_length", truncation=True, add_special_tokens=True)

            input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(device).unsqueeze(0)
            attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(device).unsqueeze(0)

            outputs = answer_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_answer_length*3)

            predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)
            result = happy_tt.generate_text("grammar: "+ predicted_answer, args=args)

            Model_Given_Answer.append(result.text)
        
        response_data = {
            'context': user_paragraph,
            'Number'  : num_questions,
            'length' : max_answer_length,
            'generated_questions': generated_questions,
            'model_answers': Model_Given_Answer
        }
        print(response_data)
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/result', methods=['POST'])
def similarity_check():
    similarity_score = []
    data = request.get_json()
    User_Entered_Answer = data.get('User_Entered_Answer', [])
    Model_Given_Answer = data.get('Model_Given_Answer', [])
    print(User_Entered_Answer)
    print(Model_Given_Answer)
    cosine_sim_spacy = []
    # Load the smaller spaCy English model
    nlp = spacy.load("en_core_web_sm")
    for i in range(len(User_Entered_Answer)):
        user_answer = User_Entered_Answer[i]
        model_answer = Model_Given_Answer[i]
        # Get embeddings for each sentence
        user_answer_embedding = nlp(str(user_answer)).vector
        model_answer_embedding = nlp(str(model_answer)).vector

        # Reshape the vectors to be 2D arrays for sklearn's cosine_similarity function
        user_answer_embedding = user_answer_embedding.reshape(1, -1)
        model_answer_embedding = model_answer_embedding.reshape(1, -1)

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(user_answer_embedding, model_answer_embedding)
        cosine_sim_spacy.append(cosine_sim[0][0])
        # Print the cosine similarity
        print(f"Cosine Similarity: {cosine_sim[0][0]}")

    Api_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX' #Generate your own API key for gemini and paste it here
    genai.configure(api_key= Api_key)
    cosine_sim_gemini = []
    for i in range(len(User_Entered_Answer)):
        embedding_sentence1_dict = genai.embed_content(
            model="models/embedding-001",
            content=str(User_Entered_Answer[i]),
            task_type="SEMANTIC_SIMILARITY"
            )

        embedding_sentence2_dict = genai.embed_content(
            model="models/embedding-001",
            content=str(Model_Given_Answer[i]),
            task_type="SEMANTIC_SIMILARITY"
            )
        embedding_sentence1 = np.array(list(embedding_sentence1_dict.values()))
        embedding_sentence2 = np.array(list(embedding_sentence2_dict.values()))

        # Reshape the vectors to be 1D arrays
        embedding_sentence1 = embedding_sentence1.reshape(-1)
        embedding_sentence2 = embedding_sentence2.reshape(-1)

        # Calculate dot product
        dot_product = np.dot(embedding_sentence1, embedding_sentence2)

        # Calculate magnitudes
        magnitude_sentence1 = np.linalg.norm(embedding_sentence1)
        magnitude_sentence2 = np.linalg.norm(embedding_sentence2)

        # Calculate cosine similarity
        cosine_similarity_score = dot_product / (magnitude_sentence1 * magnitude_sentence2)
        cosine_sim_gemini.append(cosine_similarity_score)
        print(f"Cosine Similarity Score for Answer{i+1}: {cosine_similarity_score}")
    i = 1

    for spacy_score, gemini_score in zip(cosine_sim_spacy, cosine_sim_gemini):
        print(f'Your Answer: {User_Entered_Answer[i-1]}')
        print(f'Model Answer: {Model_Given_Answer[i-1]}')
        
        i = i+1
        if gemini_score >= 0.75:
            similarity_score.append(gemini_score)
        else:
            similarity_score.append(spacy_score)
    
    similarity_score = [float(score) for score in similarity_score]

    similarity_dict = {
            'Similarity': similarity_score
        }
    return jsonify(similarity_dict)

if __name__ == "__main__":
    app.run(debug=True)
