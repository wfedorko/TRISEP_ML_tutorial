import re
import os


def print_instructions():
    usr=os.environ['USER']
    textfile = open('jupyter_logbook.txt', 'r')
    matches = []
    reg = re.compile('^\s*http://localhost:([0-9]*)')
    for line in textfile:
        match=reg.match(line)
        if match is not None:
            port=match.group(1)
            print("In a terminal running local shell on your laptop paste:")
            print('ssh -L {}:localhost:{} {}@triumf-ml1.triumf.ca -N -f'.format(port, port, usr))
            print("then in your browser address bar paste:")
            print(line.lstrip())
            
    textfile.close()




if __name__=='__main__':
    print_instructions()
