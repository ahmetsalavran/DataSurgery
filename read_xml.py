from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt
import pandas as pd
import glob
import cv2
import ast
import os
import numpy as np
path = r'C:\Users\Asus\Desktop\creating_model\datas\datas\\'
age = []
number = []
sex=[]
composition=[]
echogenicity=[]
calcifications=[]
tirads = []
images=[]
points=[]

def get_data(path,tag_name):
    content = []
    s1="<"+tag_name+">"
    s2="</"+tag_name+">"
    with open(path, "r") as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = bs(content, "lxml")
        return str(bs_content.find(tag_name)).replace(s1,"").replace(s2,"")

def get_points(path,tag_name):
    content = []
    s1="<"+tag_name+">"
    s2="</"+tag_name+">"
    with open(path, "r") as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = bs(content, "lxml")
        return ast.literal_eval(str(bs_content.find(tag_name)).replace(s1,"").replace(s2,""))[0]["points"]


def collect_data():
    for i in range(1,401):
        if os.path.exists(os.path.join(path+str(i)+"_1.jpg")):
            if get_data(path+str(i)+".xml","tirads")!="":
                image = cv2.imread(os.path.join(path+str(i)+"_1.jpg"))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image,(150,150))
                #image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
                images.append(image)
                if get_data(path+str(i)+".xml","tirads") == "2" or get_data(path+str(i)+".xml","tirads") == "3":
                    for t in range(4):
                        images.append(image)
                        tirads.append(0)
                tirads.append(1)
            else:
                continue
        else:
            continue
    return  images,np.array(tirads)


