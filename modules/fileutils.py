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
