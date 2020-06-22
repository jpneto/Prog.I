def fileMatch(filename, substr):
    f = open(filename, 'r')
    for line in f:
        if substr in line:
            yield line
 
def countLines(filename):
    f = open(filename, 'r')
    return len(f.readlines())

def countLinesBy(filename, p):
    f = open(filename, 'r')
    return len([line for line in f.readlines() if p(line)])


# use eg:

# for line in fileMatch('data/days.txt', 'sd'):
#     print(line.rstrip())
    
# print(countLines('data/days.txt'))

# print(countLinesBy('data/days.txt', lambda line : len(line)<=7))