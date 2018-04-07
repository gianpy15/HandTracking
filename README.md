# __HandTracking:__ _a deep learning project_
Our objective is to train a deep neural network able to recognize hands'
movement and provide a real time 3D recostruction of it.

## Roadmap
- [ ] [Acquire data](#Acquire-data.-A-lot-of-data.)
- [ ] [Find hand(s) in a picture](#Find-hands-in-a-picture.-Bounding-boxes.)
- [ ] [The (pure) deep learning step](#The-pure-deep-learning-step.-Training-and-prediction.)
- [ ] [The reconstruction](#The-reconstruction.)

## __Acquire data.__ _A lot of data._
The first step of each machine learning project is to find a data set.
This is the aim of this website: we shot several videos with a special camera
in order to have many frames (i.e. the pictures proposed in this website)
to be labeled. In our case the labels are all the junctions of the hand(s)
showed in the pictures.
These are required in order to train our deep neural network.</br>
![image](/web/images/labels.gif)

## __Find hands in a picture.__ _Bounding boxes._
To train the neural network we need to focus the attention on the area of
each picture where hands are present. So we need to compute the bounding
box of each sample picture.</br>
![image](/web/images/2018-03-02 21.03.29.png)

## __The pure deep learning step.__ _Training and prediction._
Finally we can train our neural network with the huge amount of data collected.</br>
![image](/web/images/cnn2.png)

## __The reconstruction.__
In the final step we plan to show a 3D real-time reconstruction of hands movement.</br>
![image](/web/images/rounding-hand.gif)

___
## About us
We are Computer Science students at Politecnico di Milano (♥) working on this project for a Deep Learning course.
Luca Cavalli, Gianpaolo Di Pietro, Michele Bertoni, Matteo Biasielli, Mattia Di Fatta
