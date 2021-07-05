# %%
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import json
import gc

# %%
#!pip install transformers 
#!pip install sentencepiece

# %%
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
    print('Neutral')
elif(prediction_value==1):
    print("Positive")
else:
    print("Negative")

print("\n");
print("\n");
# %%
print("Keywords taken by the model for prediction are")
if prediction_value==0:
  print([('flight', 2522.3073498447297), ('thanks', 966.6547381205517), ('cancelled', 952.2017229300701), ('service', 859.0770858386195), ('customer', 682.4713108923231), ('hours', 604.8896980892098), ('hold', 596.5608381486942), ('thank', 565.4297260448798), ('help', 555.1055721753867), ('delayed', 500.67842982058477), ('need', 500.1546144728829), ('flightled', 460.37157745247526), ('flights', 428.80486805163247), ('late', 386.14941582062977), ('today', 374.4010114513388), ('http', 371.3832793540942), ('airline', 354.840496248204), ('phone', 336.42287310271087), ('going', 304.634332367481), ('trying', 297.0401240803726)])
elif prediction_value==1:
  print([('http', 592.7345757040184), ('like', 393.3225496135138), ('just', 319.1590460707134), ('did', 310.7823887539117), ('guys', 309.26943322405657), ('great', 296.1974331671253), ('wait', 293.4178219499999), ('delay', 278.60439453776206), ('really', 267.4947909285881), ('good', 257.77912909310476), ('want', 242.70781325396075), ('love', 219.32862790117616), ('crew', 210.99501878557973), ('know', 199.3319473914388), ('right', 186.07437241653875), ('passengers', 174.57032713773003), ('long', 169.4807434834111), ('does', 166.68321265563344), ('better', 165.3880897434479), ('weather', 161.28368938163354)])
else:
  print([('flight', 993.3459635889899), ('time', 708.62998021564), ('just', 570.718085271489), ('gate', 463.85218865776415), ('plane', 444.11892502237737), ('bag', 440.1378277140892), ('don', 428.72477161660646), ('fly', 353.33495371171534), ('day', 303.43602104739716), ('amp', 288.20262459529806), ('change', 267.01863515166355), ('bags', 248.17652857505007), ('check', 242.43492547846856), ('luggage', 226.46531393162454), ('flying', 226.1403661698285), ('baggage', 222.57193470579844), ('lost', 219.05680041442162), ('work', 217.9663100334619), ('ticket', 201.6286756883267), ('didn', 191.61019841375298)])


# %%
