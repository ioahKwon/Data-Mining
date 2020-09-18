import math
import random
from collections import Counter

# 2015312904_JOONWOO.py
# Due Date : 2020.4.28. 오후 11시 59분
# Name: Kwon Joon Woo
# Student ID: 2015312904

# - Problem #1 : Make KNN function -
print("*************************************")
print(" # [Problem #1 : Make KNN function")
print("*************************************\n\n\n")
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
    print("\n-----------------------------------------------------------------------------\n")
    return

#------------------------------------------------------------------------------
#---------------------------------Initiation of Code---------------------------


MY_KNN(train_x1,test_x2,cl,4,2)

MY_KNN(train_x1,test_x2,cl,4,1)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#---------------------------------Problem Number 2-----------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# - Problem #2 : Make Naive Bayes Function -
print("*************************************")
print(" # [Problem #2 : Make Naive Bayes function")
print("*************************************\n\n\n")

import math

import math


#-------------------------------------------------------------------

# Make Data :

trainset_Homeowner = ['Yes','No','No','Yes','No','No','Yes','No','No','No']
trainset_Maritalstatus = ['Single','Married','Single','Married','Divorced','Married','Divorced',
                          'Single','Married','Single']
cls = ['No','No','No','No','Yes','No','No','Yes','No','Yes']

def makedata(list, name):
    trainset = []
    for i in range(0,len(list)):
        trainset.append([name,list[i]])
    return trainset


print("----------------Train Data-----------------\n")
print("Home : \n{}\n".format(makedata(trainset_Homeowner,"x1")))
print("marriage : \n{}\n".format(makedata(trainset_Maritalstatus,"x2")))
print("class : \n{}\n".format(cls))
print("-------------------------------------------\n")
home = makedata(trainset_Homeowner,"x1")
marry = makedata(trainset_Maritalstatus,"x2")

# trainset을 dic로 구현 (cls는 제외)

key = ["x1","x2"]
value = [home,marry]
select = [key,value]
dic_train = dict(zip(*select))


#-------------------------------------------------------------------
# 결과 출력용 show 함수

def show (list):
    
    i=0
    print("\n")
    for i in range(0,len(list)):
        print("     {}      ".format(list[i][0]), end = "")
        i = i +1
    print("\n")
    for i in range(0,len(list)):
        print("     {}     ".format(list[i][1]), end = "")
        i = i +1
    print("\n")

    return ""

#-------------------------------------------------------------------
def con_pro(variable,xi_value,cls_value,pri_prob,set1,set2,cls):
    # 데이터 합치기 [ID:0 ,HO:1 ,MS:2 ,Y:3]

    # 함수 이용법 예시 :
    # P(Y=no | X1=np, X2= Married)
    # pro1=con_pro('x2','Married',"No",pri_prob,x1_list,x2_list,cls)
    # print(pro1)
    #---------------------------------------------------------------
    
    temp = []
    index = 1
    
    for i in range(0,len(set1)):
        temp.append([index,set1[i],set2[i],cls[i]])
        index += 1
    
    # 데이터 정확하게 표시하기
    if variable == 'x1':
        num = 1
    elif variable == 'x2':
        num = 2
    else:
        print("Plz insert the variable name from the train set! ")
        
    #---------------------------------------
    # P(Xi=trainvalue | Y=clsvalue)

    ### Y=cls_value인 모든 경우 = y_cnt
    y_cnt = 0
    
    for i in range(0,len(temp)):
        if temp[i][-1] == cls_value:
            y_cnt += 1
        else:
            continue
    
    ### Y-clsvalue일때 Xi(variable)=xi_value일 경우의 수 = xi_cnt
    xi_cnt = 0 
    for i in range(0,len(temp)):
        class_num = len(temp[i])-1
        if xi_value == temp[i][num]:
            if cls_value == temp[i][class_num]:
                xi_cnt += 1
        else:
            continue

    ### P(Xi=xi_value | Y=cls_value)
    prob = xi_cnt/y_cnt
    return prob

#-------------------------------------------------------------------
def NaiveBayes (trainset, cls):
    #-----------------------------------------------
    # Priori Prob.

    ### 전체 클래스 몇개 있는지 확인
    new_class = [] #중복되지 않은 class 종류
    for i in range(0,len(cls)):
        if cls[i] not in new_class:
            new_class.append(cls[i])
        else:
            continue
    
    ### 전체 Y의 갯수
    total_Y = 0
    for i in new_class:
        cnt = cls.count(i)
        total_Y += cnt

    ### 전체 Y중 요소의 갯수 = 확률
    pri_prob = []
    for i in new_class:
        cnt = cls.count(i)
        prob = (cnt/total_Y)
        pri_prob.append([i,prob])
    print("Prior Prob : ",pri_prob)
    print("\n-------------------------------------------")
    #----------------------------------------------
    
    # Conditional prob.

    # 데이터 분리 (x1, x2로)
    
    keylist= trainset.keys()
    all_list = []
    x1_list = [] # x1 데이터
    x2_list = [] # x2 데이터
    
    for i in keylist:
        all_list.append(trainset[i])
    for k in keylist:
        for i in range(0,len(all_list)):
            for j in range(0,len(all_list[i])):
                if all_list[i][j][0] == k:
                    x1_list.append(all_list[i][j][1])
                elif all_list[i][j][0] != k:
                    x2_list.append(all_list[i][j][1])
                else:
                    continue
        break
    
    # cond.prob 나타내기
    
    final_x1 = ['x1']
    final_x2 = ['x2']
    final_tab = ['']
    
    for i in new_class:
        final_tab.append(i)
    print("\nConditional Probability : \n")
    print(" Below materix is shown for the sequence of {}".format(new_class))
    for i in range(0,len(new_class)):
        prob = con_pro('x1',x1_list[i],new_class[i],pri_prob,x1_list,x2_list,cls)
        final_x1.append(prob)
    print("\n\n The Final x1 : ",final_x1)

    for i in range(0,len(new_class)):
        prob = con_pro('x2',x2_list[i],new_class[i],pri_prob,x1_list,x2_list,cls)
        final_x2.append(prob)
    print("\n\n The Final x2 : ",final_x2)

    # 두 리스트 합치기
    final = final_x1+final_x2

#---------------------------------Result--------------------------------------------
    # Display
    
    print("\n\n***************Result**********************")        
    print("\nPriori Probabilities:" , pri_prob)
    print(show(pri_prob))
    print("Conditional Probabilities: \n",final)
    print("\n")
    print("I tried to display as the matrix as below...!\n")
    for x,y,z in zip(final_tab,final_x1,final_x2):
        print(x,y,z)
    print("\n\n")
# Initiation
NaiveBayes(dic_train,cls)
print("-----------------------------------------------------------------\n\n\n")


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#---------------------------------Problem Number 3-----------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# - Problem #3 : Make CART Function -

print("*************************************")
print(" # [Problem #3-1 : Make CART myInfoEntropy function")
print("*************************************\n\n\n")

# Problem 3-1

import math
#---------------------------------------------------------------------
#-------------------------------Data Set------------------------------
#---------------------------------------------------------------------

cl_1 = ['1','1','1','2','2']
cl_2 = ['1','1','2','2','2']

#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Term Frequency 찾는 Function

def termFreq(cls):
    # Distinct한 class 개수 추출
    temp =[]
    for i in cls:
        if i not in temp:
            temp.append(i)
        else:
            continue
    temp.sort()

    # 추출한 class 별로 몇개가 있는지 추출
    value = []
    
    for k in temp:
        cnt = 0
        for i in cls:
            if i == k:
                cnt += 1
            else:
                continue
        value.append(cnt)

    # 추출한 Term Frequency 정보를 딕셔너리로 합치기
    select = [temp,value]
    dic = dict(zip(*select))
    
    return dic

#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Entropy 구하는 Function

def entropy (cls,dic):
    
    # Key, Value 값 분리
    key_list = dic.keys()
    ele = []
    val = []
    for i in key_list:
        ele.append(i)
        val.append(dic[i])
    
    #----------------------------------------------
    # Entropy 계산
    #----------------------------------------------
    
    # 전체 요소 개수 구하기
    sum = 0
    cnt = 0
    for i in cls:
        cnt += 1
    total_number = cnt

    # Entropy 구하기 (1) Entropy속 Weight 구하기
    weight = []
    
    for i in ele:
        number_t = dic[i]
        p_t = number_t/total_number
        weight.append(p_t)

    # Entropy 구하기 (2) Entropy 구하기
    
    for i in range(0,len(ele)):
        ent = 0
        ent_temp = -(weight[i]*math.log2(weight[i]))
        ent += ent_temp

    return ent
#---------------------------------------------------------------------
#---------------------------------------------------------------------

def myInfoEntropy (cl_1,cl_2):
    
    # 각 split node의 Entropy 구하기
    ent_1 = entropy(cl_1,termFreq(cl_1))
    ent_2 = entropy(cl_2,termFreq(cl_2))
    
    #----------------------------------------------------------------
    # 상위 노드에서의 weight 계산
    
    # 전체 요소의 수
    sum = 0
    cnt_1 = 0 # 전체에서 1번 child node의 요소 수
    cnt_2 = 0 # 전체에서 2번 child node의 요소 수
    for i in cl_1:
        cnt_1 +=1
    for i in cl_2:
        cnt_2 +=1
    cnt = cnt_1 + cnt_2

    # weight 계산
    weight_1 = float(cnt_1/cnt) # 첫번째 노드의 전체 weight
    weight_2 = float(cnt_2/cnt) # 두번째 노드의 전체 weight

    # Alpha 구하기 (=sum.ent 구하기)
    alpha = (weight_1*ent_1) + (weight_2*ent_2)
    print("\nThe Weighted sum of information entropies is {}.\n".format(alpha))
    return alpha

#---------------------------------------------------------------------
#----------------------------Implementation---------------------------

myInfoEntropy(cl_1,cl_2)
print("--------------------------------------------------------------\n\n")
print("*************************************")
print(" # [Problem #3-2 : Make CART mySplit function")
print("*************************************\n\n\n")

#---------------------------------------------------------------------
#---------------------------------------------------------------------

# Problem 3-2

#---------------------------------------------------------------------
#-------------------------------Data Set------------------------------
#---------------------------------------------------------------------
# 제가 Python에서 pandas를 이용해 data frame을 사용할 수 있다는 것을 모르고
# Data를 dictionary로 모든 문제에 주었습니다. 최대한 비슷하게 데이터를 구성하였으나
# 조금 어설프고 오류가 있더라도 끝까지 포기하지 않고 도전한 저의 열정으로 봐주시면 감사하겠습니다.
# - 권준우 올림 -

parent_node={
    'X1': ['1','2','2','2','2','3','4','4','4','5'],
    'X2': ['4','6','5','4','3','6','6','5','4','4'],
    'Class':['1','1','1','2','2','1','1','2','2','2']
    }

cl= parent_node['Class']


#---------------------------------------------------------------------
#---------------------------------------------------------------------

def mySplit(dt,cl):
    
    # dt의 key값 분류
    key= []
    keylist = dt.keys()
    for i in keylist:
        key.append(i)

    # 딕셔너리 하나로 합치기
    value =[]
    for i in keylist:
        value.append(dt[i])

    # (X1,X2,CLASS) 이렇게 묶기
    new=[]
    for k in range(0,len(value)):
        for i in range(0,len(value[0])):
            new.append((value[k][i],value[k+1][i],value[k+2][i]))
        break
    
    # X1 Variable 정렬
    ## key 값들을 index로 정리
    cnt =0
    index_key = []
    for i in key:
        index_key.append(cnt)
        cnt+=1


    # 다시 index_key 이용하여 X1 값으로 정리
    # (X1,X2,CLASS)
    p = 1 # trial 나누는 함수 --> 표시하기 편하게 만든 것입니다.
    for k in index_key:
        # To display
        print("\n**********************[trial sort with X{}]*************************".format(p))
        p+=1
        
        # for문을 통해 X1을 기준으로 정렬하였을떄, X2를 기준으로 정렬하였을때 이런 식으로
        # Alpha를 구해서 가장 최적의 값을 찾음.
        
        new_list = sorted(new,key=lambda new:new[k]) # 정렬을 위해 요소를 Tuple 형식으로 바꿈
        new_xlist =[]
        # 다시 list로 바꿈
        for i in new_list:
            new_xlist.append(list(i))
        new_cl1 = []
        new_cl2 = []
        total_alpha =[]
        entro = [] # 각각의 엔트로피 저장할 list
        # Xi를 기준으로 두 그룹으로 나눔
        for i in range(1,(len(new_xlist)//2),1):
            list_cl1=[]
            final_cl1 =[]
            list_cl2=[]
            final_cl2 =[]
            new_cl1.append(new_xlist[:i])
            new_cl2.append(new_xlist[i+1:])
            for j in new_cl1:
                for temp in j:
                    final_cl1.append(temp)
            for j in new_cl2:
                for temp in j:
                    final_cl2.append(temp)
            real_cl1 =[]
            real_cl2 =[]
            # 나눠진 node에서 class 값 추출
            for j in final_cl1:
                real_cl1.append(j[2])
            
            for j in final_cl2:
                real_cl2.append(j[2])
                
            # 나눠진 두 개의 node에 대한 Entropy를 구해봄
            entro.append(myInfoEntropy(real_cl1,real_cl2))
            
            # 초기화
            del new_cl1 [:]
            del new_cl2 [:]
            del list_cl1 [:]
            del list_cl2 [:]
            del real_cl1 [:]
            del real_cl2 [:]
        
        print(entro)
        
        # Entropy가 가장 적은 것으로 결정
        minimum_entropy = min(entro)
        for i in range(0,len(entro)):
            if entro[i] == minimum_entropy:
                  min_index = i
                  break
            else:
                continue
        print("\nThe minimum entropy index is ",min_index)

        # index 기준으로 왼쪽 오른쪽 노드로 갈 list 찾기
        i = min_index
        dt_left =[]
        dt_right =[]
        new_cl1.append(new_xlist[:i])
        new_cl2.append(new_xlist[i+1:])
        for j in new_cl1:
            for temp in j:
                dt_left.append(temp)
        for j in new_cl2:
            for temp in j:
                dt_right.append(temp)

        # data 부분과 class 나누기
        
        dt_left_dat =[]
        dt_left_cls =[]
        dt_right_dat=[]
        dt_right_cls=[]
        
        for i in range(0,len(dt_left)):
            dt_left_dat.append([dt_left[i][0],dt_left[i][1]])
            dt_left_cls.append(dt_left[i][2])
        for i in range(0,len(dt_right)):
            dt_right_dat.append([dt_right[i][0],dt_right[i][1]])
            dt_right_cls.append(dt_right[i][2])

        # The Output
        print("\n-----# THE RESULT #-----")
        print("\nThe data frame that goes to left is : ")
        print("  - The sequence of list is : {}".format(key))
        print(dt_left)
        print("\nThe data frame that goes to right is : ")
        print("  - The sequence of list is : {}".format(key))
        print(dt_right)

    return 


#---------------------------------------------------------------------
#----------------------------Implementation---------------------------
mySplit(parent_node,cl)

# 바쁘신 와중에 제 시험지 py파일을 봐주셔서 감사합니다.
# 비록 힘들었지만 덕분에 정말 많이 배웠습니다. 감사합니다. -권준우 올림-

