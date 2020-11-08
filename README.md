# Relation Networks and Sort-of-CLEVR

GPU programming tools and algorithms have evolved dramatically over the past few years. Also, explosive increase in data had made huge development on deep-learning research. However, relational reasoning has proven difficult for neural networks to learn so far, even if it is a central component of generally intelligent behavior. To this end, I describe concise review of the paper (https://arxiv.org/abs/1706.01427) and re-implement the project from scratch.

## Descriptions

This project includes a Pytorch implementation of Relation Networks and a dataset generator which generates a synthetic VQA dataset named Sort-of-CLEVR proposed in the paper "A simple nerual network module for relational reasoning".

#### Relation Networks

<p align="center"><img src="https://github.com/ttok0s7u2n5/ML2_proj/blob/main/IMG_0182.jpg" width="500" height="250"/></p>

The above image is an illustrative example from the CLEVR dataset of relational reasoning. An image containing four objects is shown alongside non-relational and relational questions. The relational question requires explicit reasoning about the relations between the four objects in the image, whereas the non-relational question requires reasoning about the attributes of a particular object. Thinking and reasoning with relational questions is what the relation networks process.

Relation network is a neural network with a structure primed for relational reasoning. To be more specific, RN is a composite function:

<p align="center"><img src="https://github.com/ttok0s7u2n5/ML2_proj/blob/main/IMG_0183.jpg" width="300" height="135"/></p>

where the input is a set of "objects" <img src="https://render.githubusercontent.com/render/math?math=O = \{\ o_1, o_2, ...,o_n\}\\, o_i \in \mathbb{R}^m"> is the ith object, and <img src="https://render.githubusercontent.com/render/math?math=f_\phi"> and <img src="https://render.githubusercontent.com/render/math?math=g_\theta"> are functions with parameters <img src="https://render.githubusercontent.com/render/math?math=\phi"> and <img src="https://render.githubusercontent.com/render/math?math=\theta">, respectively. In the paper, the two functions are MLPs(Multi layer perceptron), and the parameters are learnable synaptic weights, making RNs end-to-end differentiable.

### Advantages of Relation Networks

#### 1. RNs learn to infer relations.

The functional form in the above equation dictates that an RN should considter the potential relations between all other object pairs. This implies that an RN is not necessarily privy to which object relations actually exist, nor to the actual meaning of any particular relation. Which means that the object of RN to learn is **to infer relations between these objects**.

#### 2. RNs are data efficient.

RNs use a single function <img src="https://render.githubusercontent.com/render/math?math=g_\theta"> to compute each relation. This can be thought of as a single function operating on a batch of object pairs. Since <img src="https://render.githubusercontent.com/render/math?math=g_\theta"> is encouraged not to over-fit to the features of any particular object pair, this mode of operation encourages **greater generalization** for computing relations.

#### 3. RNs operate on a set of objects.

From the above equation, we ensure that the RN is invariant to the order of objects in the input and it also ensures that the output is order invariant. Which means that this invariance ensures that the RN's output contains information that is generally **representative** of the relations that exist in the object set.

### Tasks

#### 1. CLEVR

Since the majority of visual QA data sets are ambiguous and exhibit strong linguistic biases, the CLEVR visual QA dataset was developed. CLEVR contains images of 3D-rendered objects and each image is associated with a number of questions that fall into different categories. CLEVR has two versions of data set: (i) the pixel version, in which images were represented in standard 2D pixel form, and (ii) a state description version, in which images were explicitly represented by state description matrices containing factored object descriptions. Each row in the matrix contained the features of a single object.

#### 2. Sort-of-CLEVR

Sort-of-CLEVR is a dataset similar to CLEVR and it separates relational and non-relational questions. It consists of images of 2D colored (total 6 colors: red, blue, green, orange, yellow, gray) shapes (square or circle) along with questions and answers about the images. This dataset is visually simple, reducing complexities involved in image processing.

#### 3. bAbI

bAbI is a pure text-based QA dataset. There are 20 tasks, each corresponding to a particular type of reasoning. Each question is associated with a set of supporting facts. For example, the facts *"Sandra picked up the football"* and *"Sandra went to the office"* support the question *"Where is the football?"*.

#### 4. Dynamic physical systems

This is a dataset of simulated physical mass-spring systems using the MuJoCo physics engine. Each scene contained 10 colored balls moving on a table-top surface. Some moved independently and other were connected by invisible springs or a rigid constraint. There're two tasks using this dataset: (i) infer the existence or absence of connections between balls when only observing their color and coordinate positions across multiple sequential frames, and (ii) count the number of systems on the table-top, again when only observing each ball's color and coordinate position across multiple sequential frames.

You can understand what it means by watching this video
: https://youtu.be/FDF6-NGv38c

### Applying Relation Network

<p align="center"><img src="https://github.com/ttok0s7u2n5/ML2_proj/blob/main/IMG_0184.jpg" width="520" height="280"/></p>

Unlike other neural network, RN (Relation Network) do not explicitly operate on images or natural language. The learning process induces upstream processing, comprised of conventional neural network modules, to produce a set of useful "objects" from distributed representations.

#### Dealing with pixels 

We would use CNN to parse pixel inputs into a set of objects, which are the images of size 128 x 128. Then, CNN convolved them through four convolutional layers to k feature maps of size d x d. Since we don't know what particular image features should consistute an object, we tag the d x d k-dimensional cells with an arbitrary coordinate indicating its relative spatial position and treat it as an object for the RN.

#### Conditioning RNs with question embeddings

Since the existence and meaning of  an object-object relation should be question dependent, we modify the RN architecture such that <img src="https://render.githubusercontent.com/render/math?math=g_\theta">could condition its processing on the question: <img src="https://render.githubusercontent.com/render/math?math=a=f_\phi(\sum_{i, j} g_\theta(o_i, o_j, q))">. Question words were assinged unique integers, which were used to index a lookup table and to get the question embedding q, we used the final state of an LSTM that processed question words.

#### Dealing with state descriptions & natural language

We provide state descriptions directly into the RN and use question processing same as before. Since bAbI dataset is only consisted of texts, we first identified up to 20 sentences in the support set. Then, we tag these sentences with labels indicating their relative position in the support set, and process each sentence word-by-word with an LSTM. Same as before, a separate LSTM produce a question embedding.

### Results

#### CLEVR from pixels

Surprisingly, the model achieved state-of-the-art performance on CLEVR at 95.5%, surpassing human performance in the task.

#### CLEVR from state descriptions

The model achieved an accuracy of 96.4%. This demostrates the generality of the RN module, which means that RNs are not necessarily restricted to visual problems and can be applied in very different contexts, and to different tasks.

#### Sort-of-CLEVR from pixels

CNN augmented with an RN achieves an accuracy above 94% for both relational and non-relational questions. But, CNN augmented with an MLP only reached this performance on the non-relational questions, which means that models lacking a dedicated relational reasoning component struggle or incapable of solving tasks that require very simple relational reasoning.

#### bAbI

The model succeeded on 18/20 tasks. It succeeded on the basic induction task and didn't catastrophically fail in any of the tasks.

#### Dynamic physical systems

In the connection inference task, the model correctly classified all the connections in 93% of the sample scenes in the test set. In the counting task, the RN achieved similar performance, reporting the correct number of connect systems for 95% of the test scene samples.

## Sort-of-CLEVR

Sort-of-CLEVR is simplified version of CLEVR.This is composed of 10000 images and 20 questions (10 relational questions and 10 non-relational questions) per each image. 6 colors (red, green, blue, orange, gray, yellow) are assigned to randomly chosen shape (square or circle), and placed in a image.

For example, with the sample image shown below, we can generate non-relational and relational questions and following answers like:

<p align="center"><img src="https://github.com/ttok0s7u2n5/ML2_proj/blob/main/0.png" width="300" height="300"/></p>

< Non-relational Questions >
1. What is the shape of the red object? => circle
2. Is green object placed on the left side of the image? => yes
3. Is gray object placed on the upside of the image? => yes

< Relational Questions >
1. What is the shape of the object closest to the red object? => circle
2. What is the shape of the object furthest to the green object? => circle
3. How many objects have same shape with the yellow object? => 4

## Setup

Install python3 (I ran these codes with 3.8.3 version) from https://www.python.org/downloads/

And install each library from requirements.txt file using pip
~~~
$ pip install -r requirements.txt
~~~

## Usage

~~~
$ ./run.sh
~~~

or

~~~
$ python sort_of_clevr_generator.py
~~~

to generate sort-or-clevr dataset and 

~~~
$ python main.py
~~~

to train the RN model. If you want to train the CNN_MLP model, then

~~~
$ python main.py --model=CNN_MLP
~~~

## Result

|  |RN|CNN_MLP|
|---|---|---|
|Non-relational question|88%|65%|
|Relational question|75%|63%|

## Reference

I modified the code more simply, referred from (https://github.com/kimhc6028/relational-networks)
