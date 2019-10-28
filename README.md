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
- 28 Oct  
>Finish train UniLM with Albert.  
>>Data Format:  
>> each line as type of 'question=answer' separated by '\n' in a file.  
>> revise config.py to specify the data folder and the directory to save the model.  

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
- <<ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS>> https://arxiv.org/pdf/1909.11942.pdf  
- <<Unified Language Model Pre-training for Natural Language Understanding and Generation>> https://arxiv.org/pdf/1905.03197.pdf