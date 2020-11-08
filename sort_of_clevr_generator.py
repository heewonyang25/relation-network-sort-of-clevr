import cv2
import os
import numpy as np
import random
import pickle
import warnings

random.seed(1)
np.random.seed(1)

train_size = 10000
test_size = 200
img_size = 75
size = 5
question_size = 18  ## 2*(6 for one-hot vector of color) + 3 for question type + 3 for question subtype
q_type_idx = 12
sub_q_type_idx = 15
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

num_questions = 10
dirs = './data'

colors = [
    (0,0,255),##r
    (0,255,0),##g
    (255,0,0),##b
    (0,156,255),##o
    (128,128,128),##k
    (0,255,255)##y
]

def translate(dataset):
    _, (rel_questions, rel_answers), (norel_questions, norel_answers) = dataset
    colors = ['red ', 'green ', 'blue ', 'orange ', 'gray ', 'yellow ']
    answer_sheet = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']

    #print("rel_questions:")
    #print(rel_questions)
    #print("rel_answers:")
    #print(rel_answers)
    print("non-relational question")
    for question,answer in zip(norel_questions,norel_answers):
        query = ''
        query += colors[question.tolist()[0:6].index(1)]
        if question[sub_q_type_idx] == 1:
            query += 'shape?'
        elif question[sub_q_type_idx+1] == 1:
            query += 'left?'
        elif question[sub_q_type_idx+2] == 1:
            query += 'up?'
        ans = answer_sheet[answer]
        print(query,'==>', ans)
    print("relational question")
    for question,answer in zip(rel_questions,rel_answers):
        query = ''
        query += colors[question.tolist()[0:6].index(1)]
        if question[sub_q_type_idx] == 1:
            query += 'closest shape?'
        elif question[sub_q_type_idx+1] == 1:
            query += 'furthest shape?'
        elif question[sub_q_type_idx+2] == 1:
            query += 'count?'
        ans = answer_sheet[answer]
        print(query,'==>', ans)

try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)        
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):
                    pas = False
        if pas:
            return center

def build_dataset():
    objects = []
    img = np.ones((img_size,img_size,3)) * 255
    for color_id,color in enumerate(colors):  
        center = center_generate(objects)
        if random.random()<0.5:
            start = (center[0]-size, center[1]-size)
            end = (center[0]+size, center[1]+size)
            cv2.rectangle(img, start, end, color, -1)
            objects.append((color_id,center,'r'))
        else:
            center_ = (center[0], center[1])
            cv2.circle(img, center_, size, color, -1)
            objects.append((color_id,center,'c'))

    rel_questions = []
    norel_questions = []
    rel_answers = []
    norel_answers = []
    
    """Non-relational questions"""
    for _ in range(num_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        norel_questions.append(question)
        """Answer : [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]"""
        if subtype == 0:
            """What is the shape of the red object? -> rectangle/circle"""
            if objects[color][2] == 'r':
                answer = 2 # if rectangle
            else:
                answer = 3 # if circle
        elif subtype == 1:
            """Is green object placed on the left side of the image? -> yes(left)/no(right)"""
            if objects[color][1][0] < img_size / 2:
                answer = 0
            else:
                answer = 1
        elif subtype == 2:
            """Is orange object placed on the upside of the image? -> yes(downside)/no(upside)"""
            if objects[color][1][1] < img_size / 2:
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)
    
    """Binary Relational questions"""
    for _ in range(num_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        rel_questions.append(question)
        """Answer : [yes, no, rectangle, circle, 1, 2, 3, 4, 5, 6]"""
        if subtype == 0:
            """What is the shape of the object closest to the red object? -> rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = 999
            closest = dist_list.index(min(dist_list))
            if objects[closest][2] == 'r':
                answer = 2
            else:
                answer = 3  
        elif subtype == 1:
            """What is the shape of the object furthest from the orange object? -> rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))
            if objects[furthest][2] == 'r':
                answer = 2
            else:
                answer = 3
        elif subtype == 2:
            """How many objects have same shape with the blue object? -> 1~6"""
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count +=1 
            answer = count+4
        rel_answers.append(answer)

    rel_relations = (rel_questions, rel_answers)
    norelations = (norel_questions, norel_answers)
    
    img = img/255.
    dataset = (img, rel_relations, norelations)
    return dataset

print('building test datasets...')
test_datasets = [build_dataset() for _ in range(test_size)]
print('building train datasets...')
train_datasets = [build_dataset() for _ in range(train_size)]

print('translating datasets...')
for i in range(train_size):
    print("{}th image".format(i))
    translate(train_datasets[i])
print('saving train dataset images to ./data')
for i in range(train_size):
    cv2.imwrite(os.path.join(dirs,'{}.png'.format(i)), cv2.resize(train_datasets[i][0]*255, (512,512)))

print('saving datasets...')
filename = os.path.join(dirs,'sort-of-clevr.pickle')
with open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets), f)
print('datasets saved at {}'.format(filename))
