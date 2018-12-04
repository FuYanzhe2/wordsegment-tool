# coding=utf-8
import jpype
import os.path
import sys, time
import os

if __name__ == "__main__":
    # 打开jvm虚拟机
    
    jar_path = os.path.abspath('/home/fyz/nlp/hlseg/5.1.13/lib')
    jvmPath = jpype.getDefaultJVMPath()
    # 如果系统里安装了多个版本的java,则可以直接用下面的语句指定jvm版本
    #	jvmPath = "/usr/local/jdk/jdk1.8.0_162/jre/lib/amd64/server/libjvm.so"
    jpype.startJVM(jvmPath, "-ea", "-Djava.class.path=%s" % (jar_path + '/hlSegment-5.1.13.jar'),
                   "-Djava.ext.dirs=%s" % jar_path)

    # 取得类定义
    BasicSegmentor = jpype.JClass('com.hylanda.segmentor.BasicSegmentor')
    SegOption = jpype.JClass('com.hylanda.segmentor.common.SegOption')
    SegGrain = jpype.JClass('com.hylanda.segmentor.common.SegGrain')
    SegResult = jpype.JClass('com.hylanda.segmentor.common.SegResult')

    # 创建分词对象
    segmentor = BasicSegmentor()


    # 加载词典, 如果有用户自定义词典，可以用下面的语句加载，需要注意文件码制
    if not segmentor.loadDictionary("/home/fyz/nlp/hlseg/5.1.13/dictionary/CoreDict.dat", None):
        #	if not segmentor.loadDictionary("./dictionary/CoreDict.dat", "./dictionary/userDict_utf8.txt"):
        print("字典加载失败！")
        exit()

    # 创建SegOption对象，如果使用默认的分词选项，也可以直接传空
    option = SegOption()
    option.mergeNumeralAndQuantity = False
    # 可以使用下面的语句调整分词颗粒度SMALL,NORMAL,LARGE
    option.grainSize = SegGrain.LARGE
    #pos
    option.doPosTagging = True
    # 分词
    
    segResult = segmentor.segment("去美国要保证金的都是旅行社行为怕你黑在那里，要担责任，自由行是不用的。", option)

    # 遍历分词结果
    word = segResult.getFirst()
    words = segResult.getKeywordsList()
    result = ""
    while (word != None):
        result += word.wordStr +"/"+str(word.nature)+" "
        word = word.next


    # 输出结果
    print(result)
    jpype.shutdownJVM()
    exit()
