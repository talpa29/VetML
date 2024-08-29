# from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline, AutoModelForSeq2SeqLM, BertForQuestionAnswering, BertTokenizer
# import torch


# # # Load pre-trained model and tokenizer
# # model_name = "VetBERT"
# # model_name = "C:/Users/Tal Partosh/Desktop/Project/Datasets/archive/vetbert.ckpt"


# # tokenizer = BertTokenizer.from_pretrained(model_name)
# # model = BertForSequenceClassification.from_pretrained(model_name)


# # # Prepare input data
# # text = "My dog ​​has a fever of 39.5 degrees Celsius and has no appetite for the last day what could be the reason?"
# # inputs = tokenizer(text, return_tensors='pt')

# # # Get model predictions
# # with torch.no_grad():
# #     outputs = model(**inputs)
# #     logits = outputs.logits

# # # Convert logits to predicted class
# # predicted_class = torch.argmax(logits, dim=1)

# # tokenizer = AutoTokenizer.from_pretrained("havocy28/VetBERT")
# # model = AutoModelForMaskedLM.from_pretrained("havocy28/VetBERT")


# dash_line = '-'.join('' for x in range(100))

# tokenizer = AutoTokenizer.from_pretrained("havocy28/VetBERT")
# model = AutoModelForMaskedLM.from_pretrained("havocy28/VetBERT")

# # VetBERT_masked = pipeline("question-answering", model=model, tokenizer=tokenizer)
# # # question_one = VetBERT_masked('Suspected pneuomina, will require an [MASK] but in the meantime will prescribed antibiotics')
# # context = "My dog ​​has a fever of 39.5 degrees Celsius and has no appetite for the last day."
# # question = "What could be the reason?"
# # qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
# # result = qa_pipeline(question=question, context=context)

# # question_two = VetBERT_masked('My dog ​​has a fever of 39.5 degrees Celsius and has no appetite for the last day what could be the reason?')
# # print(question_one)
# # print(question_two)
# # model_name = 'VetBERT'

# tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
# model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
# qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
# context = "My dog ​​has a fever of 39.5 degrees Celsius and has no appetite for the last day."
# question = "What could be the reason?"
# result = qa_pipeline(question=question, context=context)
# res = result['answer']
# # question = 'My dog ​​has a fever of 39.5 degrees Celsius and has no appetite for the last day what could be the reason?'
# # inputs = tokenizer(question, return_tensors='pt')
# # output = tokenizer.decode(
# #     model.generate(
# #         inputs["input_ids"], 
# #         max_new_tokens=50,
# #     )[0], 
# #     skip_special_tokens=True
# # )
# print(dash_line)
# print('Example 1:')
# print('INPUT DIALOGUE:\nSuspected pneuomina, will require an [MASK] but in the meantime will prescribed antibiotics')
# print(f'OUTPUT:\n{res}')
# print(dash_line)
# # print('Example 2:')
# # print('INPUT DIALOGUE:\nMy dog ​​has a fever of 39.5 degrees Celsius and has no appetite for the last day what could be the reason?')
# # print(f'OUTPUT:\n{question_two}')
# # print(dash_line)


from transformers import BertTokenizer, BertForQuestionAnswering, pipeline

def main():
    # Load the VetBERT tokenizer and QA model 
    # tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    # model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    tokenizer = BertTokenizer.from_pretrained("havocy28/VetBERT")
    model = BertForQuestionAnswering.from_pretrained("havocy28/VetBERT")
    # Initialize the QA pipeline with the VetBERT model
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Define the context and the question
    context = """
    A dog has a subcutaneous mass in the abdominal area.
    """
    question = "should we try to take a sample using a test syringe in case it's a cyst, or do an x-ray to make sure it's not intestines?"

    # Get the answer from the model
    result = qa_pipeline(question=question, context=context)

    print("Question:", question)
    print("Answer:", result['answer'])

if __name__ == '__main__':
    main()
