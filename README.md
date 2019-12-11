# BERT_TF
produced by Andysin Zhang  

*Many years ago the great British explorer George Mallory, who was to die on Mount Everest, was asked why did he want to climb it. He said, "Because it is there."
Well, space is there, and we're going to climb it, and the moon and the planets are there, and new hopes for knowledge and peace are there. And, therefore, as we set sail we ask God's blessing on the most hazardous and dangerous and greatest adventure on which man has ever embarked.*			------President John F. Kennedy

This package will be the foundation on which our platform would be based on, it will be very difficult, however, **DO NOT LET THE FAILURE DEFINE US, LET FAILURE TEACH US !**  

## Requirements  
- python 3  
- tensorflow == 1.14 

## Run example  
```shell
python pre_train.py
```

## Log  
- 11 Dec  
>>1.Be going to rewrite all the code.  
>>2.Add VAE in model_helper.py.  

- 04 Nov  
>>Revise the `load_data.py`, enhance the capability of generation.  

- 01 Nov  
>>Successfully restore pre-trained bert model, please create a directory which saves the pre-trained model,
>>the directory of pre-trained model should not be the same as the model save path, and change the init_checkpoint in the config.py  

>>Need to ameliorate:  
>>>>1.When do predict, the maximum length should be identical to the training steps, need to fix this;  
>>>>2.add `start`, `end` tag to the data;  

- 30 Oct
>>The model now could be run correctly.  
>>add `run_predict.py` for prediction, however, due to the training process could see the answer,  
>>the result is not good when do prediction, maybe should train three language models where the original paper mentions.  

- 28 Oct  
>Finish train UniLM with Albert.  
>>Data Format:  
>>>> each line as type of 'question=answer' separated by '\n' in a file.  
>>>> revise config.py to specify the data folder and the directory to save the model.  

>> TODO:
>>>>1.finish evalutaion metrics; 2.save model as pb in order to deploy on the server; 3. write the interface for serving input;  

- 22 Oct  
>>Move the previous version, which use tf.placeholder to receive the data, to the desperated directory.  
>>New model use tf.estimator, tf.Dataset, and could reload the pre-train Bert model easily.  
>>Finish Writing the BERT Model, and ALBERT.

- 18 Oct  
>>for tf.estimator, tf.Dataset, write a brief but of complete structure tutorial, see tutorial directory.  

- 14th Oct  
>>~~finish UniLM, need to rewrite infer()~~     

- 25th Sep  
>>~~decide to implement UNI-LM~~  

- 19th Sep  
>>~~finish fully train procedure.~~    

- 18th Sep  
>>~~finish pre-train Bert, write train example.~~   

>>~~TODO: <1>data utils; ~~<2>fully train procedure;~~ <3>fine tune tasks;~~   

## Reference  

- TensorFlow code and pre-trained models for BERT https://arxiv.org/abs/1810.04805  
- https://github.com/google-research/bert  
- << ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS>> https://arxiv.org/pdf/1909.11942.pdf  
- << Unified Language Model Pre-training for Natural Language Understanding and Generation>> https://arxiv.org/pdf/1905.03197.pdf