import re
import sys  

##reload(sys)  
##sys.setdefaultencoding('utf8')

#with open("movie.txt", "r") as titles:
arr = []
with open("movie_titles.txt", "r",encoding ='ISO-8859-1') as titles: 
    for title in titles:
##        title = str(title, 'utf-8')
        title = title.rstrip()
        title = title.split(',')[2]
        arr.append(title)
print(len(arr))

cluster = []   
with open("test1.txt", "r") as fp:
    for line in fp:
        sub = []
        rr = re.findall(r'\d+' ,line)
        for i in range(len(rr)):
           if (rr[i].isdigit()):
               index = int(rr[i])
               sub.append(index)
        if(sub != [] and len(sub) >= 2):
            cluster.append(sub)

outputfile = open("moviematch.txt", "w")
print(len(cluster))
with open('moviematch.txt', 'w') as outputfile:
    result = []
    for i in range(len(cluster)):
        subresult = []
        for j in range(len(cluster[i])):
            outputfile.write(arr[cluster[i][j]])
            outputfile.write(" , ")
            subresult.append(arr[cluster[i][j]])
        result.append(subresult)
        outputfile.write("\n-----\n")
    outputfile.close()

##with open("output.txt", "r",encoding ='ISO-8859-1') as outputfile:
##    print(outputfile.readlines())
####print(result)





        



                
##
##        print(line)

