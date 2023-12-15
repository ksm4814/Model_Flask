import numpy as np
import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizer

# 저장된 토크나이저를 불러옵니다.
bert_tokenizer = BertTokenizer.from_pretrained('./klue_roberta')
bert_finetuning = torch.load(
    './klue_roberta_model_97.pth', map_location=torch.device('cpu'))

app = Flask(__name__)

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        # Flutter에서 전송한 데이터를 받습니다.
        data = request.get_json(force=True)
        text = data['text']

        # 받은 텍스트에 대해 감성 분석을 수행합니다.
        tokenized_text = bert_tokenizer.encode_plus(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            output = bert_finetuning(**tokenized_text)
            logits = output.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # 감성 분석 결과를 전송합니다.
        emotions = ['0', '1', '2', '3', '4', '5']
        result = {'emotion': emotions[predicted_class]}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
