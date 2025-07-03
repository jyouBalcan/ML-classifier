import random



def balance(x, y):

    # Count samples per category & record minimum count
    s = set(y)
    c = set()
    m = len(x)
    for a in s:
        temp = y.count(a)
        if temp < m:
            m = temp
            min_category = a
        c.add(temp)
        
    #if not the same length, we pad with shuffled entries
    if len(c) > 1:
        
        count = 0
        i = 0
        while count < max(c)-m:
            if y[i] == min_category:

                #shuffle
                entry = x[i].split()
                random.shuffle(entry)
                entry = ' '.join(entry)

                #add to list
                y.append(min_category)
                x.append(entry)
                count += 1
            i += 1
            #update counts
        return balance(x,y)
            
    
    return x,y

   
