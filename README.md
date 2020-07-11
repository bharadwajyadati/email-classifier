Email signature processing requires two models working in sequence to produce the signature

1) Preprocessing needs to be done to split emails into promotions, updates and inbox (like in Gmail) , manual preprocessing step is tedious
and requires blacklisting for each client and the list keeps growing as we see and then pandas dataframe needs to exclude few subjects , few senders etc ..

Above step can be modelled using a machine learning model ? to filter out all the mails and give only mails from inbox 

Machine learning models to try out , simple svm ? evaluate with accuracy else replace with rnn

2) After we get inbox mails , needs to have an automated system for reading the body and then finding out signature from it , if exists

Attention based seqtoseq model for the same




https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb


https://pytorch.org/tutorials/beginner/transformer_tutorial.html



https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/



installation of lightgbm


brew install lightgbm