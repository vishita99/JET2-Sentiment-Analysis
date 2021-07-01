from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import json
import gc

save_directory = "model/"                            #Giving the path of the models
        
test_text = input("Please give your feedback...")    #taking input from the user
print(test_text)
loaded_tokenizer = DistilBertTokenizer.from_pretrained(save_directory)                          #Loading the tokenizer
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(save_directory)            #Loadng the model
predict_input = loaded_tokenizer.encode(test_text,truncation=True,padding=True,return_tensors="tf")   #encoding the text
output = loaded_model(predict_input)[0]                                         #predicting using the given model
prediction_value = tf.argmax(output, axis=1).numpy()[0]                         #taking the maximum probability of the all the probabs

print("*"*100)
if(prediction_value==0):                                           #matching with the encoded text that we did at the start of the model
    print('Neagtive')
elif(prediction_value==1):
    print("Neutral")
else:
    print("Positive")
