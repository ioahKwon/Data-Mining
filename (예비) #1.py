import math
import random
from collections import Counter

# 2015312904_JOONWOO.py
# Due Date : 2020.4.28. 오후 11시 59분

# - Problem #1 : Make KNN function -

#--------------------------Traning example set---------------------------------

train_x1 = [2,3,4,5,5,11] # 임의의 Train set
test_x2 = [7,4,6,1,3,8] # 임의의 Test set
cl = ['T','A','F','A','F',"T"] # 임의의 class 값 (Train set의 class임)

#--------------------------------Main Code-------------------------------------

# 전체 KNN 함수 

def MY_KNN(trainset, testset, cl, k, method):
    
    print("--------------------------------------")
    print("\n*** KNN function is intialized ***\n")
    print("--------------------------------------")
    
    # Distance -> (코드설명 1)

    print("1 : Function for calculating Distance is activated \n")
    print("Procesing....")
    print("Finished! \n")

    # Find Kth Nearest Neighbors -> (코드설명 2)

    print("2 : Function for finding Kth nearest Neighbors \n")
    print("Procesing....")
    kth = find_kth_big(distance(trainset,testset),k)
    print("Finished! \n")

    # Classification -> (코드설명 3)
    
    print("3 : Function for Classification \n")
    print("Procesing....")
    print("Finished! \n")
    find_class(kth,distance(trainset,testset),cl,method,testset)
    
    return

#------------------------------------------------------------------------------

#--------------------------------코드설명 1------------------------------------

# Distance 구하는 함수 - testset 기준으로 train set 까지 거리
def distance(trainset, testset):
    
    # 변수 선언
    temp = []
    newlist=[]
    
    # distance 구하기
    for k in range(len(testset)):
        for j in range(len(trainset)):
            sum = 0
            sum = sum + ((testset[k]-trainset[j])**2)
            dist = abs(sum**0.5)
            temp.append(dist)
    
    # 리스트 안 요소 쪼개기
    n = len(trainset)
    for i in range(0,len(temp),n):
        result = newlist.append(temp[i:i+n])
    return newlist

#------------------------------------------------------------------------------

#--------------------------------코드설명 2------------------------------------

# Kth Nearest Neighbor를 구하는 함수 - 거리상 가장 가까운 k개의 sample 추출
def find_kth_big(list,k):
    list_max = 0
    max_temp = []
    for i in range(0,len(list)):
        list[i].sort(reverse=False)
        max_temp.append(list[i][0:k])
    return max_temp

#------------------------------------------------------------------------------

#--------------------------------코드설명 3------------------------------------

# 사용자가 지정한 Method에 따라 Classification 하는 과정

def find_class(kth_list,ori_dis_list,cls,method,test_list):
    
    # 변수 선언
    temp = 0
    tempo = [] # knn의 index
    temp_list = []
    class_list =[] ## knn의 index인 tempo를 가지고 유추한 Trainset의 class들
    value_tempo = []
    final_class_major = [] # Method 1의 classification 결과
    final_class_weight = []

    # 거리가 가까운 순으로 index 추출
    for i in range(0,len(kth_list)):
        for j in range(0,len(kth_list[i])):
            temp_list = ori_dis_list[temp]
            value = temp_list.index(kth_list[i][j])
            
            # index가 temporary list에 없으면 추가/ value도 같이 다른 list에 저장
            if value not in tempo:
                tempo.append(value)
                value_tempo.append(temp_list[value])

            # index가 temporary list에 있으면 중복이니 다음 번호 찾는 과정
            else:
                value_tempo.append(temp_list[value])
                temp_list[value] = 'disabled' # 값을 바꿔주지 않으면  중복된 값이 다시 들어가므로
                new_value = temp_list.index(kth_list[i][j])
                tempo.append(new_value)
        
        # Testset을 통해 추출한 KNN을 이용하여 Trainingset의 class 유추
        for k in tempo:
            class_list.append(cls[k])
            
        # Method 1: Majority voting
        if method == 1:
            max = 0
            for k in class_list:
                current = class_list.count(k)
                if max < current :
                    max = current
                    sol = []
                    sol.append(k)
                elif max == current :
                    sol.append(k)
                else:
                    continue
                
            # Final Class for majority vote.
            variable = random.choice(sol) #동점이면 Random하게 뽑음
            final_class_major.append(variable)

        # Method 2: Distance-Weighted Voting
        elif method == 2:
            inv_dis = [] # 거리의 역수
            
            # 거리 역수 구하기
            for i in range(0,len(value_tempo)):
                variable1 = value_tempo[i]
                if variable1 == 0:
                    inverse = 0
                    inv_dis.append(inverse)
                else:
                    inverse = 1/(value_tempo[i])
                    inv_dis.append(inverse)
            
            # 중복된 CLASS 있으면 weight값 합쳐주기
            
            list_t = []
            for i in range(0,len(class_list)):
                list_t.append([class_list[i],inv_dis[i]])
            
            for i in range(0,len(list_t)-2):
                
                if list_t[i][0] == list_t[i+1][0]:
                    sum = list_t[i][1] + list_t[i+1][1]
                    list_t.remove(list_t[i])
                    list_t[i][1] = sum
                elif list_t[1][0] == list_t[-1][0]:
                    sum = list_t[1][1] + list_t[-1][1]
                    list_t.remove(list_t[1])
                    list_t[-1][1] = sum
                    
                elif list_t[0][0] == list_t[-1][0]:
                    sum = list_t[-1][1] + list_t[0][1]
                    list_t.remove(list_t[0])
                    list_t[-1][1] = sum
                    
                elif list_t[0][0] == list_t[-2][0]:
                    sum = list_t[-2][1] + list_t[0][1]
                    list_t.remove(list_t[0])
                    list_t[-2][1] = sum
                    
                elif list_t[2][0] == list_t[i][0]:
                    sum = list_t[i][1] + list_t[2][1]
                    list_t.remove(list_t[2])
                    list_t[i][1] = sum
                
                else:
                    continue

            # 중복 제외한 데이터 토대로 classification
            
            temp_values =[]
            for k in range(0,len(list_t)):
                temp_values.append(list_t[k][1])

            # temp_values 중 가장 큰 수 찾기
            largest = temp_values[0]
            for i in temp_values:
                if i > largest:
                    largest = i
            max_=largest
            
            for i in range(0,len(list_t)):
                if max_ == list_t[i][1]:
                    number = list_t[i][0]
                    final_class_weight.append(number)
            
            # Initialization
            del value_tempo [:]

        # Method 1,2가 아닌 다른 숫자가 들어갔을때
        else:
            print("You put the wrong method number. Please try again")
             
        temp+=1

        # 초기화
        del tempo [:]
        del class_list[:]
        

    #-----------------------------------------------------
    # Final determination of Class according to the method
    #-----------------------------------------------------
    
    print("\n------------------------------RESULTS-----------------------------------\n")
    if method == 1:
        final_class = final_class_major
    elif method == 2:
        final_class = final_class_weight
    else:
        print("Error! Revise Method number")
        
    # Making dictionary to Display
    
    Test_value = test_list
    class_ = final_class

            
    select = [Test_value,class_]
    dic = dict(zip(*select))
    print("*** The predicted class of the Test data ***\n")
    i =1
    print("Predicted class was extracted based on method {}".format(method))
    for Test_value, class_ in dic.items():
        print("Final Result {} : (Test_Value, Predicted class) = ({} : {})".format(i,Test_value,class_))
        i+=1
    print("\n*************************************************************************\n")
    return

#------------------------------------------------------------------------------
#---------------------------------Initiation of Code---------------------------


MY_KNN(train_x1,test_x2,cl,4,2)

MY_KNN(train_x1,test_x2,cl,4,1)
