from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext-large")