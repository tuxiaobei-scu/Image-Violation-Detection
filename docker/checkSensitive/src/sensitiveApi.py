from __future__ import division

import re

MinMatchType = 1 #最小匹配规则，如：敏感词库["中国", "中国人"]，语句："我是中国人"，匹配结果：我是[中国]人
MaxMatchType = 2 #最大匹配规则，如：敏感词库["中国", "中国人"]，语句："我是中国人"，匹配结果：我是[中国人]


def getReg(txt_convert):
    """
    对文本进行正则过滤，检测广告、链接等信息
    :param txt: 文本
    :return: 正则过滤后的文本
    """
    url_patten = r"([^\s]+(\.com))|([a-zA-z]+://[^\s]*)" #http://xxx, www.xxxx.com, 1234@qq.com
    html_patten=r"<(\S*?)[^>]*>.*?|<.*? />"
    qq_phone_patten=r"[1-9][0-9]{4,}" #第一位1-9之间的数字，第二位0-9之间的数字，大于1000号
    wx_patten=r"[a-zA-Z][a-zA-Z0-9_-]{5,19}$"

    if re.findall(url_patten,txt_convert).__len__()>0:
        result = u"疑似[网页链接或邮箱]"
    elif re.findall(html_patten,txt_convert).__len__()>0:
        result = u"疑似[html脚本]"
    elif re.findall(qq_phone_patten,txt_convert).__len__()>0:
        result = u"疑似[QQ号或手机号]"
    elif re.findall(wx_patten,txt_convert).__len__()>0:
        result = u"疑似[微信号]"
    else:
        result = u"非广告文本"
    return result



def calcScore(sensitiveWordStr):
    b=sensitiveWordStr
    b1=b.split(",")
    b2=[i.split(":")[0] for i in b1 if len(i) > 1]

    score = 0
    for x in b2:
        if x in (u"毒品", u"色情", u"赌博"):
            score += 5
        elif x in (u"政治", u"反动", u"暴恐"):
            score += 4
        elif x == u"社会":
            score += 3
        else: #其他
            score += 2
    return score



def calcGrade(score,sensitive_list_word_length,txt_length):
    if score>15 and sensitive_list_word_length/txt_length>=0.33:
        suggest=u"删除"
    elif score==0:
        suggest=u"通过"
    else:
        suggest=u"掩码"
    return suggest


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""

    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += chr(inside_code)
    return rstring


class DFAFilter():
    def __init__(self):
        # 特殊字符集
        with open("checkSensitive/data/stopword.txt", encoding='utf-8') as f:
            self.stopWordSet = [i.split('\n')[0] for i in f.readlines()]

        # 敏感词集
        with open("checkSensitive/data/dict.txt", encoding='utf-8') as f1:
            lst = f1.readlines()
            self.sensitiveWordSet = [i.split("\n")[0].split("\t") for i in lst]

        self.sensitiveWordMap = self.initSensitiveWordMap(self.sensitiveWordSet)

    def initSensitiveWordMap(self, sensitiveWordSet):
        """
        初始化敏感词库，构建DFA算法模型
        :param sensitiveWordSet: 敏感词库,包括词语和其对应的敏感类别
        :return: DFA模型
        """
        sensitiveWordMap=dict()
        for category,key in sensitiveWordSet:
            nowMap = sensitiveWordMap
            for i in range(len(key)):
                keyChar =key[i]  # 转换成char型
                wordMap = nowMap.get(keyChar) #库中获取关键字
                #如果存在该key，直接赋值，用于下一个循环获取
                if wordMap != None:
                    nowMap =wordMap
                else:
                    #不存在则构建一个map，同时将isEnd设置为0，因为不是最后一个
                    newWorMap = dict()
                    #不是最后一个
                    newWorMap["isEnd"]="0"
                    nowMap[keyChar]=newWorMap
                    nowMap = newWorMap
                #最后一个
                if i ==len(key)-1:
                    nowMap["isEnd"]="1"
                    nowMap["category"]=category
        return sensitiveWordMap


    def checkSensitiveWord(self, txt,beginIndex,matchType=MinMatchType):
        """
        检查文字中是否包含敏感字符
        :param txt:待检测的文本
        :param beginIndex: 调用getSensitiveWord时输入的参数，获取词语的上边界index
        :param matchType:匹配规则 1：最小匹配规则，2：最大匹配规则
        :return:如果存在，则返回敏感词字符的长度，不存在返回0
        """
        flag=False
        category=""
        matchFlag=0  #敏感词的长度
        nowMap=self.sensitiveWordMap
        tmpFlag=0  #包括特殊字符的敏感词的长度

        # print "len(txt)",len(txt) #9
        for i in range(beginIndex,len(txt)):
            word = txt[i]

            #检测是否是特殊字符，eg"法&&轮&功..."
            if word in self.stopWordSet and len(nowMap)<100:
                #len(nowMap)<100 保证已经找到这个词的开头之后出现的特殊字符
                #eg"情节中,法&&轮&功..."这个逗号不会被检测
                tmpFlag += 1
                continue


            #获取指定key
            nowMap=nowMap.get(word)
            if nowMap !=None: #存在，则判断是否为最后一个
                #找到相应key，匹配标识+1
                matchFlag+=1
                tmpFlag+=1
                #如果为最后一个匹配规则，结束循环，返回匹配标识数
                if nowMap.get("isEnd")=="1":
                    #结束标志位为true
                    flag=True
                    category=nowMap.get("category")
                    #最小规则，直接返回,最大规则还需继续查找
                    if matchType==MinMatchType:
                        break
            else: #不存在，直接返回
                break


        if matchFlag<2 or not flag: #长度必须大于等于1，为词
            tmpFlag=0
        return tmpFlag,category


    def contains(self, txt,matchType=MinMatchType):
        """
        判断文字是否包含敏感字符
        :param txt: 待检测的文本
        :param matchType: 匹配规则 1：最小匹配规则，2：最大匹配规则
        :return: 若包含返回true，否则返回false
        """
        flag=False
        for i in range(len(txt)):
            matchFlag=self.checkSensitiveWord(txt,i,matchType)[0]
            if matchFlag>0:
                flag=True
        return flag


    def getSensitiveWord(self, txt,matchType=MinMatchType):
        """
        获取文字中的敏感词
        :param txt: 待检测的文本
        :param matchType: 匹配规则 1：最小匹配规则，2：最大匹配规则
        :return:文字中的敏感词
        """
        sensitiveWordList=list()
        for i in range(len(txt)): #0---11
            length = self.checkSensitiveWord(txt, i, matchType)[0]
            category=self.checkSensitiveWord(txt, i, matchType)[1]
            if length>0:
                word=txt[i:i + length]
                sensitiveWordList.append(category+":"+word)
                i = i + length - 1
        return sensitiveWordList


    def replaceSensitiveWord(self, txt, replaceChar, matchType=MinMatchType):
        """
        替换敏感字字符
        :param txt: 待检测的文本
        :param replaceChar:用于替换的字符，匹配的敏感词以字符逐个替换，如"你是大王八"，敏感词"王八"，替换字符*，替换结果"你是大**"
        :param matchType: 匹配规则 1：最小匹配规则，2：最大匹配规则
        :return:替换敏感字字符后的文本
        """
        tupleSet = self.getSensitiveWord(txt, matchType)
        wordSet=[i.split(":")[1] for i in tupleSet]
        resultTxt=""
        if len(wordSet)>0: #如果检测出了敏感词，则返回替换后的文本
            for word in wordSet:
                replaceString=len(word)*replaceChar
                txt = txt.replace(word, replaceString)
                resultTxt=txt
        else: #没有检测出敏感词，则返回原文本
            resultTxt = txt
        return resultTxt


    def detect(self, txt):
        txt_length=len(txt)
        txt_convert= strQ2B(txt) #全角转半角
        reg_result= getReg(txt_convert) #正则过滤

        if reg_result==u"非广告文本":
            #是否包含敏感词
            contain = self.contains(txt=txt_convert,matchType=MaxMatchType) #默认 MinMatchType

            #敏感词和其类别
            sensitive_list = self.getSensitiveWord(txt=txt_convert, matchType=MaxMatchType)  #默认 MinMatchType
            sensitive_list_str=u','.join(sensitive_list) #字符串形式的敏感词和其类别
            sensitive_list_word=[i.split(":")[1] for i in sensitive_list] #敏感词

            #敏感词的字数
            sensitive_list_word_length=0
            for word in sensitive_list_word :
                if len(word)<=1:
                    continue
                sensitive_list_word_length+=len(word)

            #待检测语句的敏感度得分
            score= calcScore(sensitive_list_str)
            #待检测语句的敏感级别
            grade= calcGrade(score, sensitive_list_word_length, txt_length)
            #替换敏感词后的文本
            txt_replace=self.replaceSensitiveWord(txt=txt_convert,replaceChar='*',matchType=MaxMatchType) #默认MinMatchTYpe

            result_json={
                u"txt":txt,
                u"txtLength":txt_length,
                u"regularResult":reg_result,
                u"ifContainSensitiveWord":contain,
                u"sensitiveWordCount":len(sensitive_list),
                u"sensitiveWordList":"["+sensitive_list_str+u"]",
                u"score":score,
                u"grade":grade,
                u"txtReplace":txt_replace
            }

        else:
            result_json={
                u"txt":txt,
                u"txtLength":txt_length,
                u"regularResult":reg_result,
                u"grade":u"删除"
            }


        return result_json

