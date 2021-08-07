import numpy as np
import sys
import matplotlib.pyplot as plt
import os
def find_ntm_events(time,y,threshold=10,scheme=None):
   res=[]
   assert(len(time)==len(y))
   if len(time)==0:
      return []

   if threshold==None:
      if scheme==None:
         print(np.mean(y))
         threshold=np.mean(y)*3
   print(threshold)
   previous_end = -100
   during_elm = False
   current_elm={}
   for i,yi in enumerate(y):
       if yi>threshold:
          if during_elm == False:
             if time[i]-previous_end>100 or len(res)==0:
                current_elm['begin']=time[i]
                current_elm['max']=yi
                during_elm=True
                print('detected NTM at',time[i],'ms')
             else:
                #print('Combining two ELM crashes......')
                current_elm=res.pop()
                during_elm=True
                current_elm['max']=yi
                
          else:
             current_elm['max'] = max(yi,current_elm['max'])
       else:
          if during_elm == True:
             during_elm = False
             current_elm['end']=time[i]
             res.append(current_elm)
             current_elm={}
             previous_end=time[i]
   #          print('ELM ended at',time[i],'ms')
    #         print('******************************************')
   if during_elm == True:
             during_elm = False
             current_elm['end']=time[i]
             res.append(current_elm)
             current_elm={}
             previous_end=time[i]

    
   print(len(res),'NTM events detected~~~!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

   return res

