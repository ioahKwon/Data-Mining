import math

############## myinfoEntropy 설계하는 문제


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

######################################################################
######################################################################
######################################################################
######################################################################

############## mySplit 설계하는 문제


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






















