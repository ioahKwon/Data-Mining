# Problem 2
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
    for x,y,z in zip(final_tab,final_x1,final_x2):
        print(x,y,z)
        
# Initiation
NaiveBayes(dic_train,cls)
