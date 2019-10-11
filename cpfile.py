import os
import xml.etree.ElementTree as et
import glob
import filecmp as fc
import shutil as cp
import random as rd
import tkinter as tk
def dup(num):
    for i in a:
      if i == num:
        return True
    return False
def cp_img(fol,des):
  filenames_xml = glob.glob(fol + "\\*.xml")

  filenames_jpg = glob.glob(fol + "\\*.jpg")

  count = 0
  global a
  a = []
  for xml in filenames_xml:
     count = count + 1
  #print('Tong file xml: ',count)
  count2 = round(count*80/100)
  #print(count2)
  for i in range(count2):
    rand = rd.randint(1,count)
    #print('1st',rand) 
    while(dup(rand)):
      rand = rd.randint(1,count)
    #print('2st',rand) 
    cp.copy(filenames_xml[rand-1],des+'\\train')
    cp.copy(filenames_jpg[rand-1],des+'\\train')
    #print('copy')
    a.append(rand)
  for y in range(count-count2):
    rand = rd.randint(1,count)
    while(dup(rand)):
      rand = rd.randint(1,count)
    cp.copy(filenames_xml[rand-1],des+'\\test')
    cp.copy(filenames_jpg[rand-1],des+'\\test')
    #print('copy')
    a.append(rand)
  

'''k = 0
z = 0
for z in range(count2):
  z = z + 1
for train in trains:
  k = k + 1
print('xml: ', k)
print('count2: ', z)'''


  
 
  


