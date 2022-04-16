
#LSTM

LSTM refers to as Long Short Term Memory. Sequence Prediction Problems have been encountered a lot in this new Tech driven society. It includes anything from recommending new words in sentences to recommending new songs. LSTM is found out to be one of the most effective solution compared to RNN(Recurrent Neural Network) and CNN(convolution neural Network) .Starting from CNN ,it took the test cases to be independent and only used the current input to predict the outcome.and RNN technique has a shorter duration of memory which makes it a harder for sequence 
prediction.

![image](https://user-images.githubusercontent.com/55499361/163659240-23ad94db-8ba5-4e38-b8a4-500c44c7229f.png)

#Architecture of LSTMs
An LSTM model comprises of various different blocks called as cells.It generally consists of two states which are being transferred to the next cell ,i.e the cell state and hidden state. The memory blocks or cell are in charge of remembering things, and they can be manipulated through three basic mechanisms known as gates.

A)#Forget Gate
In this Gate ,when this point is reached the information which is no longer further required in the process any further is dropped out or discarded.It helps a lot in optimizing LTSM network

B)#INPUT GATE

The input gate is responsible for the addition of information to the current cell state.

C)#Output Gate

Output Gate job is of selecting useful information from the current cell state and showing it out as an output.

